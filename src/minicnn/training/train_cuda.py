#!/usr/bin/env python3
"""Train a small VGG-style CUDA CNN on CIFAR-10.

Architecture:
Conv(3->32) -> LeakyReLU -> Conv(32->32) -> LeakyReLU -> MaxPool
Conv(32->64) -> LeakyReLU -> Conv(64->64) -> LeakyReLU -> MaxPool
FC(1600->10)
"""
import os
import time
from ctypes import c_float
from pathlib import Path

import numpy as np

import minicnn.config.settings as S
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
from minicnn.paths import DATA_ROOT, ARTIFACTS_ROOT, BEST_MODELS_ROOT

RUN_DIR = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
RUN_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = str(BEST_MODELS_ROOT / f"{RUN_DIR.name}_{S.BEST_MODEL_FILENAME}")


def _conv_backward_step(
    n, log_grad,
    d_grad_in_nchw,   # NCHW gradient from upstream layer
    d_grad_cnhw,      # buffer for CNHW gradient (pool-cnhw when maxpool present, else conv-raw-grad)
    d_conv_raw,       # pre-relu conv output (for leaky_relu_backward)
    d_col,            # im2col columns reused from forward pass
    d_conv_input,     # input to this conv layer (for weight gradient)
    d_weight, d_w_grad, d_upstream_grad, d_velocity,
    c_in, in_h, in_w, c_out, out_h, out_w,
    lr, name,
    d_conv_raw_grad=None, d_max_idx=None, pool_h=None, pool_w=None,
):
    """Run one conv layer's backward pass: (optional maxpool→) relu → conv → weight update."""
    if d_max_idx is not None:
        nchw_to_cnhw_into(d_grad_in_nchw, d_grad_cnhw, n, c_out, pool_h, pool_w)
        zero_floats(d_conv_raw_grad, c_out * n * out_h * out_w)
        lib.maxpool_backward_use_idx(d_grad_cnhw, d_max_idx, d_conv_raw_grad, n, c_out, out_h, out_w)
        lib.leaky_relu_backward(d_conv_raw, d_conv_raw_grad, c_float(S.LEAKY_ALPHA), c_out * n * out_h * out_w)
        d_back = d_conv_raw_grad
    else:
        nchw_to_cnhw_into(d_grad_in_nchw, d_grad_cnhw, n, c_out, out_h, out_w)
        lib.leaky_relu_backward(d_conv_raw, d_grad_cnhw, c_float(S.LEAKY_ALPHA), c_out * n * out_h * out_w)
        d_back = d_grad_cnhw
    lib.conv_backward_precol(
        d_back, d_conv_input, d_weight, d_w_grad, d_upstream_grad, d_col,
        n, c_in, in_h, in_w, S.KH, S.KW, out_h, out_w, c_out,
    )
    update_conv(
        d_weight, d_w_grad, d_velocity, lr,
        S.MOMENTUM, c_out * c_in * S.KH * S.KW, name,
        S.WEIGHT_DECAY, S.GRAD_CLIP_CONV,
        out_h * out_w if S.CONV_GRAD_SPATIAL_NORMALIZE else 1.0,
        log_grad,
    )


def main():
    data_root = str(DATA_ROOT)
    x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_cifar10(
        data_root,
        n_train=S.N_TRAIN,
        n_val=S.N_VAL,
        seed=S.DATASET_SEED,
        train_batch_ids=S.TRAIN_BATCH_IDS,
    )
    x_train = normalize_cifar(x_train)
    x_val = normalize_cifar(x_val)
    x_test_final = normalize_cifar(x_test_final)
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test(official): {x_test_final.shape[0]}")

    w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b = init_weights(S.INIT_SEED)
    d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = upload_weights(
        w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b
    )
    d_v_conv1, d_v_conv2, d_v_conv3, d_v_conv4, d_v_fc_w, d_v_fc_b = init_velocity_buffers()

    print(
        "Arch: Conv1(3->32)->Conv2(32->32)->Pool1"
        f"->Conv3(32->64)->Conv4(64->64)->Pool2->FC({S.FC_IN}->10)"
    )
    print(
        f"Shapes: 32x32 -> {S.H1}x{S.W1} -> {S.H2}x{S.W2} -> {S.P1H}x{S.P1W}"
        f" -> {S.H3}x{S.W3} -> {S.H4}x{S.W4} -> {S.P2H}x{S.P2W}"
    )
    print(
        f"LR_conv1={S.LR_CONV1}, LR_conv={S.LR_CONV}, LR_fc={S.LR_FC}, "
        f"momentum={S.MOMENTUM}, weight_decay={S.WEIGHT_DECAY}, BATCH={S.BATCH}, EPOCHS={S.EPOCHS}"
    )
    print(f"DATASET_SEED={S.DATASET_SEED}, INIT_SEED={S.INIT_SEED}, TRAIN_SEED={S.TRAIN_SEED}")
    print()

    NBATCHES = (x_train.shape[0] + S.BATCH - 1) // S.BATCH
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    plateau_count = 0
    current_lr_conv1 = S.LR_CONV1
    current_lr_conv = S.LR_CONV
    current_lr_fc = S.LR_FC
    workspace = BatchWorkspace()
    train_rng = np.random.default_rng(S.TRAIN_SEED)

    def _device_weights():
        return DeviceWeights(d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b)

    def _velocity_buffers():
        return VelocityBuffers(d_v_conv1, d_v_conv2, d_v_conv3, d_v_conv4, d_v_fc_w, d_v_fc_b)

    try:
        for epoch in range(S.EPOCHS):
            t0 = time.time()
            total_loss = 0.0
            correct = 0
            total_seen = 0
            indices = train_rng.permutation(x_train.shape[0])

            for batch_idx in range(NBATCHES):
                log_grad = S.GRAD_DEBUG and batch_idx < S.GRAD_DEBUG_BATCHES
                idx_s = batch_idx * S.BATCH
                idx_e = min(idx_s + S.BATCH, x_train.shape[0])
                n = idx_e - idx_s
                if n <= 0:
                    continue
                x = x_train[indices[idx_s:idx_e]]
                y = y_train[indices[idx_s:idx_e]]

                x = augment_batch(x, train_rng, S.RANDOM_CROP_PADDING, S.HORIZONTAL_FLIP)

                # ── Forward pass (keep intermediates for backward) ──────────
                upload_to(workspace.d_x, x)
                upload_int_to(workspace.d_y, y)

                conv_forward_into(workspace.d_x, d_w_conv1, workspace.d_col1, workspace.d_conv1_raw, n, S.C1_IN, S.H, S.W, S.C1_OUT)
                cnhw_to_nchw_into(workspace.d_conv1_raw, workspace.d_conv1_nchw, n, S.C1_OUT, S.H1, S.W1)

                conv_forward_into(workspace.d_conv1_nchw, d_w_conv2, workspace.d_col2, workspace.d_conv2_raw, n, S.C2_IN, S.H1, S.W1, S.C2_OUT)
                maxpool_forward_into(workspace.d_conv2_raw, workspace.d_pool1, workspace.d_max_idx1, n, S.C2_OUT, S.H2, S.W2)
                cnhw_to_nchw_into(workspace.d_pool1, workspace.d_pool1_nchw, n, S.C2_OUT, S.P1H, S.P1W)

                conv_forward_into(workspace.d_pool1_nchw, d_w_conv3, workspace.d_col3, workspace.d_conv3_raw, n, S.C3_IN, S.P1H, S.P1W, S.C3_OUT)
                cnhw_to_nchw_into(workspace.d_conv3_raw, workspace.d_conv3_nchw, n, S.C3_OUT, S.H3, S.W3)

                conv_forward_into(workspace.d_conv3_nchw, d_w_conv4, workspace.d_col4, workspace.d_conv4_raw, n, S.C4_IN, S.H3, S.W3, S.C4_OUT)
                maxpool_forward_into(workspace.d_conv4_raw, workspace.d_pool2, workspace.d_max_idx2, n, S.C4_OUT, S.H4, S.W4)
                cnhw_to_nchw_into(workspace.d_pool2, workspace.d_pool2_nchw, n, S.C4_OUT, S.P2H, S.P2W)

                lib.dense_forward(workspace.d_pool2_nchw, d_fc_w, d_fc_b, workspace.d_fc_out, n, S.FC_IN, 10)
                lib.gpu_memset(workspace.d_loss_sum, 0, 4)
                lib.gpu_memset(workspace.d_correct, 0, 4)
                lib.softmax_xent_grad_loss_acc(
                    workspace.d_fc_out, workspace.d_y, workspace.d_probs,
                    workspace.d_grad_logits, workspace.d_loss_sum, workspace.d_correct,
                    n, 10,
                )
                batch_loss_sum = download_float_scalar(workspace.d_loss_sum)
                batch_correct = download_int_scalar(workspace.d_correct)
                # Kernel stores batch loss as a sum; divide by sample count for mean loss.
                total_loss += batch_loss_sum
                correct += batch_correct
                total_seen += n

                # ── Backward pass ───────────────────────────────────────────
                lib.dense_backward_full(
                    workspace.d_grad_logits, workspace.d_pool2_nchw, d_fc_w,
                    workspace.d_pool2_grad_nchw, workspace.d_fc_grad_w, workspace.d_fc_grad_b,
                    n, S.FC_IN, 10,
                )
                lib.conv_update_fused(
                    d_fc_w, workspace.d_fc_grad_w, d_v_fc_w,
                    c_float(current_lr_fc), c_float(S.MOMENTUM), c_float(S.WEIGHT_DECAY),
                    c_float(S.GRAD_CLIP_FC), c_float(1.0), 10 * S.FC_IN,
                )
                lib.conv_update_fused(
                    d_fc_b, workspace.d_fc_grad_b, d_v_fc_b,
                    c_float(current_lr_fc), c_float(S.MOMENTUM), c_float(0.0),
                    c_float(S.GRAD_CLIP_BIAS), c_float(1.0), 10,
                )

                lib.clip_inplace(workspace.d_pool2_grad_nchw, c_float(S.GRAD_POOL_CLIP), n * S.FC_IN)
                _conv_backward_step(
                    n, log_grad,
                    workspace.d_pool2_grad_nchw, workspace.d_pool2_grad,
                    workspace.d_conv4_raw, workspace.d_col4, workspace.d_conv3_nchw,
                    d_w_conv4, workspace.d_w_conv4_grad, workspace.d_conv3_grad, d_v_conv4,
                    S.C4_IN, S.H3, S.W3, S.C4_OUT, S.H4, S.W4,
                    current_lr_conv, "conv4",
                    d_conv_raw_grad=workspace.d_conv4_grad_raw, d_max_idx=workspace.d_max_idx2,
                    pool_h=S.P2H, pool_w=S.P2W,
                )
                _conv_backward_step(
                    n, log_grad,
                    workspace.d_conv3_grad, workspace.d_conv3_grad_raw,
                    workspace.d_conv3_raw, workspace.d_col3, workspace.d_pool1_nchw,
                    d_w_conv3, workspace.d_w_conv3_grad, workspace.d_pool1_grad, d_v_conv3,
                    S.C3_IN, S.P1H, S.P1W, S.C3_OUT, S.H3, S.W3,
                    current_lr_conv, "conv3",
                )
                _conv_backward_step(
                    n, log_grad,
                    workspace.d_pool1_grad, workspace.d_pool1_grad_cnhw,
                    workspace.d_conv2_raw, workspace.d_col2, workspace.d_conv1_nchw,
                    d_w_conv2, workspace.d_w_conv2_grad, workspace.d_conv1_grad, d_v_conv2,
                    S.C2_IN, S.H1, S.W1, S.C2_OUT, S.H2, S.W2,
                    current_lr_conv, "conv2",
                    d_conv_raw_grad=workspace.d_conv2_grad_raw, d_max_idx=workspace.d_max_idx1,
                    pool_h=S.P1H, pool_w=S.P1W,
                )
                _conv_backward_step(
                    n, log_grad,
                    workspace.d_conv1_grad, workspace.d_conv1_grad_raw,
                    workspace.d_conv1_raw, workspace.d_col1, workspace.d_x,
                    d_w_conv1, workspace.d_w_conv1_grad, workspace.d_x_grad, d_v_conv1,
                    S.C1_IN, S.H, S.W, S.C1_OUT, S.H1, S.W1,
                    current_lr_conv1, "conv1",
                )

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Batch {batch_idx+1}/{NBATCHES}: loss={batch_loss_sum/n:.4f}, batch_acc={batch_correct/n*100:.1f}%")

            if hasattr(lib, 'gpu_synchronize'):
                lib.gpu_synchronize()
            train_acc = correct / total_seen * 100
            val_acc = evaluate(x_val, y_val, _device_weights())
            epoch_loss = total_loss / total_seen
            elapsed = time.time() - t0
            improved = val_acc > (best_val_acc + S.MIN_DELTA)

            if improved:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0
                plateau_count = 0
                save_checkpoint(
                    BEST_MODEL_PATH, epoch + 1, val_acc,
                    current_lr_conv1, current_lr_conv, current_lr_fc,
                    _device_weights(),
                )
                save_msg = " [saved best]"
            else:
                epochs_no_improve += 1
                plateau_count += 1
                save_msg = ""

            if plateau_count >= S.LR_PLATEAU_PATIENCE:
                new_lr_conv1 = max(current_lr_conv1 * S.LR_REDUCE_FACTOR, S.MIN_LR)
                new_lr_conv = max(current_lr_conv * S.LR_REDUCE_FACTOR, S.MIN_LR)
                new_lr_fc = max(current_lr_fc * S.LR_REDUCE_FACTOR, S.MIN_LR)
                if (new_lr_conv1, new_lr_conv, new_lr_fc) != (current_lr_conv1, current_lr_conv, current_lr_fc):
                    current_lr_conv1 = new_lr_conv1
                    current_lr_conv = new_lr_conv
                    current_lr_fc = new_lr_fc
                    print(
                        f"  LR reduced -> conv1={current_lr_conv1:.6f}, "
                        f"conv={current_lr_conv:.6f}, fc={current_lr_fc:.6f}"
                    )
                plateau_count = 0

            print(
                f"Epoch {epoch+1}/{S.EPOCHS}: Loss={epoch_loss:.4f}, Train={train_acc:.2f}%, Val={val_acc:.2f}%, "
                f"BestVal={best_val_acc:.2f}% @ {best_epoch}, "
                f"LRs=({current_lr_conv1:.6f}, {current_lr_conv:.6f}, {current_lr_fc:.6f}), "
                f"Time={elapsed:.1f}s{save_msg}"
            )

            if epochs_no_improve >= S.EARLY_STOP_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs; best val {best_val_acc:.2f}% at epoch {best_epoch}.")
                break

    finally:
        workspace.free()

    print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
    free_weights(_velocity_buffers())
    if os.path.exists(BEST_MODEL_PATH):
        best_ckpt, fc_w, fc_b, new_device_weights = reload_weights_from_checkpoint(
            BEST_MODEL_PATH, _device_weights(),
        )
        d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = new_device_weights
        print(f"Reloaded best checkpoint from epoch {int(best_ckpt['epoch'])} with Val={float(best_ckpt['val_acc']):.2f}%")
    test_acc = evaluate(x_test_final, y_test_final, _device_weights())
    print(f"Test Accuracy: {test_acc:.2f}%")

    free_weights(_device_weights())
    print("\nDone!")


if __name__ == "__main__":
    main()
