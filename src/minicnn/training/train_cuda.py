#!/usr/bin/env python3
"""Train a VGG-style CUDA CNN on CIFAR-10.

Architecture is read from the YAML `model.conv_layers` list; adding or
removing layers requires only a YAML edit — no Python changes.
"""
import os
import time
from ctypes import c_float

import numpy as np

from minicnn.data.cifar10 import load_cifar10, normalize_cifar
from minicnn.core.cuda_backend import (
    download_float_scalar,
    download_int_scalar,
    lib,
    update_conv,
)
from minicnn.training.evaluation import evaluate
from minicnn.models.initialization import init_weights
from minicnn.training.checkpoints import (
    DeviceWeights,
    VelocityBuffers,
    free_weights,
    init_velocity_buffers,
    reload_weights_from_checkpoint,
    save_checkpoint,
    upload_weights,
)
from minicnn.training.cuda_ops import (
    cnhw_to_nchw_into,
    conv_forward_into,
    maxpool_forward_into,
    nchw_to_cnhw_into,
    upload_int_to,
    upload_to,
    zero_floats,
)
from minicnn.training.cuda_epoch import augment_batch
from minicnn.training.cuda_workspace import BatchWorkspace
from minicnn.config.settings import (
    BATCH,
    BEST_MODEL_FILENAME,
    CONV_GRAD_SPATIAL_NORMALIZE,
    DATASET_SEED,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_FC,
    GRAD_DEBUG,
    GRAD_DEBUG_BATCHES,
    GRAD_POOL_CLIP,
    INIT_SEED,
    LEAKY_ALPHA,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    LR_PLATEAU_PATIENCE,
    LR_REDUCE_FACTOR,
    MIN_DELTA,
    MIN_LR,
    MOMENTUM,
    N_TRAIN,
    N_VAL,
    RANDOM_CROP_PADDING,
    HORIZONTAL_FLIP,
    TRAIN_BATCH_IDS,
    TRAIN_SEED,
    WEIGHT_DECAY,
    get_arch,
)

from pathlib import Path
from minicnn.paths import DATA_ROOT, ARTIFACTS_ROOT, BEST_MODELS_ROOT

RUN_DIR = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
RUN_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = str(BEST_MODELS_ROOT / f"{RUN_DIR.name}_{BEST_MODEL_FILENAME}")

# Module-level GPU pointer state — set in main() and used by helpers below.
_d_weights: list = []
_d_velocities: list = []
_d_fc_w = None
_d_fc_b = None
_d_v_fc_w = None
_d_v_fc_b = None


def current_device_weights() -> DeviceWeights:
    return DeviceWeights(_d_weights, _d_fc_w, _d_fc_b)


def current_velocity_buffers() -> VelocityBuffers:
    return VelocityBuffers(_d_velocities, _d_v_fc_w, _d_v_fc_b)


def _prev_nchw(i: int, ws: BatchWorkspace) -> object:
    """NCHW input pointer for stage i (the NCHW output of stage i-1, or d_x)."""
    if i == 0:
        return ws.d_x
    s_prev = get_arch().conv_stages[i - 1]
    return ws.d_pool_nchw[i - 1] if s_prev.pool else ws.d_conv_nchw[i - 1]


def main() -> None:
    global _d_weights, _d_velocities, _d_fc_w, _d_fc_b, _d_v_fc_w, _d_v_fc_b

    arch = get_arch()

    data_root = str(DATA_ROOT)
    x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_cifar10(
        data_root, n_train=N_TRAIN, n_val=N_VAL,
        seed=DATASET_SEED, train_batch_ids=TRAIN_BATCH_IDS,
    )
    x_train = normalize_cifar(x_train)
    x_val   = normalize_cifar(x_val)
    x_test_final = normalize_cifar(x_test_final)
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test_final.shape[0]}")

    *conv_arrays, fc_w, fc_b = init_weights(INIT_SEED, arch)
    dw = upload_weights(conv_arrays, fc_w, fc_b)
    _d_weights    = dw.conv_weights
    _d_fc_w, _d_fc_b = dw.fc_w, dw.fc_b
    vb = init_velocity_buffers(arch)
    _d_velocities = vb.conv_velocities
    _d_v_fc_w, _d_v_fc_b = vb.fc_w_vel, vb.fc_b_vel

    stage_str = ' -> '.join(
        f"Conv({s.in_c}→{s.out_c}){'+Pool' if s.pool else ''}"
        for s in arch.conv_stages
    )
    print(f"Arch: {stage_str} -> FC({arch.fc_in}→{arch.fc_out})")
    print(f"LR_conv1={LR_CONV1}, LR_conv={LR_CONV}, LR_fc={LR_FC}, "
          f"momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, BATCH={BATCH}, EPOCHS={EPOCHS}")
    print()

    NBATCHES = (x_train.shape[0] + BATCH - 1) // BATCH
    best_val_acc   = -1.0
    best_epoch     = -1
    epochs_no_improve = 0
    plateau_count  = 0
    current_lr_conv1 = LR_CONV1
    current_lr_conv  = LR_CONV
    current_lr_fc    = LR_FC
    workspace = BatchWorkspace()
    train_rng = np.random.default_rng(TRAIN_SEED)

    try:
        for epoch in range(EPOCHS):
            t0 = time.time()
            total_loss = 0.0
            correct = 0
            total_seen = 0
            indices = train_rng.permutation(x_train.shape[0])

            for batch_idx in range(NBATCHES):
                log_grad = GRAD_DEBUG and batch_idx < GRAD_DEBUG_BATCHES
                idx_s = batch_idx * BATCH
                idx_e = min(idx_s + BATCH, x_train.shape[0])
                n = idx_e - idx_s
                if n <= 0:
                    continue
                x = augment_batch(
                    x_train[indices[idx_s:idx_e]],
                    train_rng, RANDOM_CROP_PADDING, HORIZONTAL_FLIP,
                )
                upload_to(workspace.d_x, x)
                upload_int_to(workspace.d_y, y_train[indices[idx_s:idx_e]])

                # ---- Forward pass ----
                for i, s in enumerate(arch.conv_stages):
                    conv_forward_into(
                        _prev_nchw(i, workspace), _d_weights[i],
                        workspace.d_col[i], workspace.d_conv_raw[i],
                        n, s.in_c, s.h_in, s.w_in, s.out_c,
                    )
                    if s.pool:
                        maxpool_forward_into(
                            workspace.d_conv_raw[i], workspace.d_pool[i],
                            workspace.d_max_idx[i], n, s.out_c, s.h_out, s.w_out,
                        )
                        cnhw_to_nchw_into(
                            workspace.d_pool[i], workspace.d_pool_nchw[i],
                            n, s.out_c, s.ph, s.pw,
                        )
                    else:
                        cnhw_to_nchw_into(
                            workspace.d_conv_raw[i], workspace.d_conv_nchw[i],
                            n, s.out_c, s.h_out, s.w_out,
                        )

                last_s = arch.conv_stages[-1]
                fc_in_ptr = workspace.d_pool_nchw[-1] if last_s.pool else workspace.d_conv_nchw[-1]

                lib.dense_forward(fc_in_ptr, _d_fc_w, _d_fc_b,
                                  workspace.d_fc_out, n, arch.fc_in, arch.fc_out)
                lib.gpu_memset(workspace.d_loss_sum, 0, 4)
                lib.gpu_memset(workspace.d_correct, 0, 4)
                lib.softmax_xent_grad_loss_acc(
                    workspace.d_fc_out, workspace.d_y,
                    workspace.d_probs, workspace.d_grad_logits,
                    workspace.d_loss_sum, workspace.d_correct,
                    n, arch.fc_out,
                )
                total_loss += download_float_scalar(workspace.d_loss_sum)
                correct    += download_int_scalar(workspace.d_correct)
                total_seen += n

                # ---- FC backward + weight update ----
                lib.dense_backward_full(
                    workspace.d_grad_logits, fc_in_ptr, _d_fc_w,
                    workspace.d_pre_fc_grad_nchw,
                    workspace.d_fc_grad_w, workspace.d_fc_grad_b,
                    n, arch.fc_in, arch.fc_out,
                )
                lib.conv_update_fused(
                    _d_fc_w, workspace.d_fc_grad_w, _d_v_fc_w,
                    c_float(current_lr_fc), c_float(MOMENTUM), c_float(WEIGHT_DECAY),
                    c_float(GRAD_CLIP_FC), c_float(1.0), arch.fc_out * arch.fc_in,
                )
                lib.conv_update_fused(
                    _d_fc_b, workspace.d_fc_grad_b, _d_v_fc_b,
                    c_float(current_lr_fc), c_float(MOMENTUM), c_float(0.0),
                    c_float(GRAD_CLIP_BIAS), c_float(1.0), arch.fc_out,
                )

                # ---- Conv backward loop (reverse order) ----
                lib.clip_inplace(workspace.d_pre_fc_grad_nchw,
                                 c_float(GRAD_POOL_CLIP), n * arch.fc_in)
                grad_nchw = workspace.d_pre_fc_grad_nchw

                for i in reversed(range(arch.n_conv)):
                    s = arch.conv_stages[i]

                    if s.pool:
                        nchw_to_cnhw_into(grad_nchw, workspace.d_pool_grad_cnhw[i],
                                          n, s.out_c, s.ph, s.pw)
                        zero_floats(workspace.d_conv_raw_grad[i],
                                    s.out_c * n * s.h_out * s.w_out)
                        lib.maxpool_backward_use_idx(
                            workspace.d_pool_grad_cnhw[i], workspace.d_max_idx[i],
                            workspace.d_conv_raw_grad[i], n, s.out_c, s.h_out, s.w_out,
                        )
                    else:
                        nchw_to_cnhw_into(grad_nchw, workspace.d_conv_raw_grad[i],
                                          n, s.out_c, s.h_out, s.w_out)

                    lib.leaky_relu_backward(
                        workspace.d_conv_raw[i], workspace.d_conv_raw_grad[i],
                        c_float(LEAKY_ALPHA), s.out_c * n * s.h_out * s.w_out,
                    )
                    lib.conv_backward_precol(
                        workspace.d_conv_raw_grad[i], _prev_nchw(i, workspace),
                        _d_weights[i], workspace.d_w_grad[i],
                        workspace.d_input_nchw_grad[i], workspace.d_col[i],
                        n, s.in_c, s.h_in, s.w_in, s.kh, s.kw,
                        s.h_out, s.w_out, s.out_c,
                    )

                    lr = current_lr_conv1 if i == 0 else current_lr_conv
                    spatial_norm = s.h_out * s.w_out if CONV_GRAD_SPATIAL_NORMALIZE else 1.0
                    update_conv(
                        _d_weights[i], workspace.d_w_grad[i], _d_velocities[i],
                        lr, MOMENTUM, s.weight_numel,
                        f"conv{i + 1}", WEIGHT_DECAY, GRAD_CLIP_CONV if i < arch.n_conv else GRAD_CLIP_FC,
                        spatial_norm, log_grad,
                    )

                    grad_nchw = workspace.d_input_nchw_grad[i]

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Batch {batch_idx+1}/{NBATCHES}: "
                          f"loss={total_loss/total_seen:.4f}, "
                          f"acc={correct/total_seen*100:.1f}%")

            if hasattr(lib, 'gpu_synchronize'):
                lib.gpu_synchronize()
            train_acc = correct / total_seen * 100
            val_acc   = evaluate(x_val, y_val, current_device_weights())
            epoch_loss = total_loss / total_seen
            elapsed = time.time() - t0
            improved = val_acc > (best_val_acc + MIN_DELTA)

            if improved:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0
                plateau_count = 0
                save_checkpoint(
                    BEST_MODEL_PATH, epoch + 1, val_acc,
                    current_lr_conv1, current_lr_conv, current_lr_fc,
                    current_device_weights(), arch,
                )
                save_msg = " [saved best]"
            else:
                epochs_no_improve += 1
                plateau_count += 1
                save_msg = ""

            if plateau_count >= LR_PLATEAU_PATIENCE:
                new_lr_conv1 = max(current_lr_conv1 * LR_REDUCE_FACTOR, MIN_LR)
                new_lr_conv  = max(current_lr_conv  * LR_REDUCE_FACTOR, MIN_LR)
                new_lr_fc    = max(current_lr_fc    * LR_REDUCE_FACTOR, MIN_LR)
                if (new_lr_conv1, new_lr_conv, new_lr_fc) != (current_lr_conv1, current_lr_conv, current_lr_fc):
                    current_lr_conv1, current_lr_conv, current_lr_fc = new_lr_conv1, new_lr_conv, new_lr_fc
                    print(f"  LR → conv1={current_lr_conv1:.6f}, "
                          f"conv={current_lr_conv:.6f}, fc={current_lr_fc:.6f}")
                plateau_count = 0

            print(
                f"Epoch {epoch+1}/{EPOCHS}: Loss={epoch_loss:.4f}, "
                f"Train={train_acc:.2f}%, Val={val_acc:.2f}%, "
                f"BestVal={best_val_acc:.2f}% @ {best_epoch}, "
                f"LRs=({current_lr_conv1:.6f},{current_lr_conv:.6f},{current_lr_fc:.6f}), "
                f"Time={elapsed:.1f}s{save_msg}"
            )

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs; "
                      f"best val {best_val_acc:.2f}% at epoch {best_epoch}.")
                break

    finally:
        workspace.free()

    print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
    free_weights(current_velocity_buffers())
    if os.path.exists(BEST_MODEL_PATH):
        best_ckpt, fc_w, fc_b, new_dw = reload_weights_from_checkpoint(
            BEST_MODEL_PATH, current_device_weights(), arch,
        )
        _d_weights = new_dw.conv_weights
        _d_fc_w, _d_fc_b = new_dw.fc_w, new_dw.fc_b
        print(f"Reloaded best checkpoint from epoch {int(best_ckpt['epoch'])} "
              f"with Val={float(best_ckpt['val_acc']):.2f}%")
    test_acc = evaluate(x_test_final, y_test_final, current_device_weights())
    print(f"Test Accuracy: {test_acc:.2f}%")
    free_weights(current_device_weights())
    print("\nDone!")


if __name__ == "__main__":
    main()
