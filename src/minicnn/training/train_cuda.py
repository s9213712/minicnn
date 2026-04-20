#!/usr/bin/env python3
"""Train a VGG-style CUDA CNN on CIFAR-10.

Architecture is read from the YAML `model.conv_layers` list; adding or
removing layers requires only a YAML edit — no Python changes.
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from minicnn.training.cuda_arch import CudaNetGeometry

from minicnn.training.evaluation import evaluate
from minicnn.models.initialization import init_weights
from minicnn.training.checkpoints import (
    free_weights,
    init_adam_buffers,
    init_velocity_buffers,
    reload_weights_from_checkpoint,
    save_checkpoint,
    upload_weights,
)
from minicnn.training.cuda_batch import CudaRuntimeState, synchronize_if_available, train_cuda_batch
from minicnn.training.cuda_epoch import augment_batch
from minicnn.training.cuda_workspace import BatchWorkspace
from minicnn.training.legacy_data import load_normalized_cifar10
from minicnn.training.loop import (
    EpochTimer,
    FitState,
    LrState,
    RunningMetrics,
    format_epoch_summary,
    reduce_lr_on_plateau,
)
from minicnn.config.settings import (
    BATCH,
    BEST_MODEL_FILENAME,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GRAD_DEBUG,
    GRAD_DEBUG_BATCHES,
    INIT_SEED,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    LR_PLATEAU_PATIENCE,
    LR_REDUCE_FACTOR,
    MIN_DELTA,
    MIN_LR,
    MOMENTUM,
    OPTIMIZER_TYPE,
    RANDOM_CROP_PADDING,
    HORIZONTAL_FLIP,
    TRAIN_SEED,
    WEIGHT_DECAY,
    get_arch,
)

from minicnn.paths import ARTIFACTS_ROOT, BEST_MODELS_ROOT

RUN_DIR = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
RUN_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = str(BEST_MODELS_ROOT / f"{RUN_DIR.name}_{BEST_MODEL_FILENAME}")


def init_cuda_runtime(arch: "CudaNetGeometry") -> CudaRuntimeState:
    *conv_arrays, fc_w, fc_b = init_weights(INIT_SEED, arch)
    use_adam = OPTIMIZER_TYPE.lower() == 'adam'
    ln_stages = [s for s in arch.conv_stages if s.layer_norm]
    ln_gamma = [np.ones(s.out_c, dtype=np.float32) for s in ln_stages]
    ln_beta  = [np.zeros(s.out_c, dtype=np.float32) for s in ln_stages]
    return CudaRuntimeState(
        weights=upload_weights(conv_arrays, fc_w, fc_b, ln_gamma, ln_beta),
        velocities=init_velocity_buffers(arch),
        adam=init_adam_buffers(arch) if use_adam else None,
    )


def print_training_header(arch: "CudaNetGeometry", x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> None:
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")
    stage_str = ' -> '.join(
        f"Conv({s.in_c}→{s.out_c}){'+Pool' if s.pool else ''}"
        for s in arch.conv_stages
    )
    print(f"Arch: {stage_str} -> FC({arch.fc_in}→{arch.fc_out})")
    print(f"LR_conv1={LR_CONV1}, LR_conv={LR_CONV}, LR_fc={LR_FC}, "
          f"momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, BATCH={BATCH}, EPOCHS={EPOCHS}")
    print()


def run_cuda_epoch(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_rng: np.random.Generator,
    lr_state: LrState,
) -> RunningMetrics:
    metrics = RunningMetrics()
    indices = train_rng.permutation(x_train.shape[0])
    nbatches = (x_train.shape[0] + BATCH - 1) // BATCH

    for batch_idx in range(nbatches):
        idx_s = batch_idx * BATCH
        idx_e = min(idx_s + BATCH, x_train.shape[0])
        n = idx_e - idx_s
        if n <= 0:
            continue

        x = augment_batch(
            x_train[indices[idx_s:idx_e]],
            train_rng, RANDOM_CROP_PADDING, HORIZONTAL_FLIP,
        )
        y = y_train[indices[idx_s:idx_e]]
        train_cuda_batch(
            runtime,
            workspace,
            arch,
            x,
            y,
            lr_state,
            metrics,
            log_grad=GRAD_DEBUG and batch_idx < GRAD_DEBUG_BATCHES,
        )

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{nbatches}: "
                  f"loss={metrics.loss:.4f}, "
                  f"acc={metrics.acc_percent:.1f}%")

    return metrics


def save_best_checkpoint(epoch: int, val_acc: float, lr_state: LrState, runtime: CudaRuntimeState, arch: "CudaNetGeometry") -> None:
    save_checkpoint(
        BEST_MODEL_PATH, epoch, val_acc,
        lr_state.conv1, lr_state.conv, lr_state.fc,
        runtime.weights, arch,
    )


def reduce_lr_if_due(fit: FitState, lr_state: LrState) -> None:
    if reduce_lr_on_plateau(fit, lr_state, LR_PLATEAU_PATIENCE, LR_REDUCE_FACTOR, MIN_LR):
        print(f"  LR -> conv1={lr_state.conv1:.6f}, "
              f"conv={lr_state.conv:.6f}, fc={lr_state.fc:.6f}")


def run_final_evaluation(runtime: CudaRuntimeState, arch: "CudaNetGeometry", x_test: np.ndarray, y_test: np.ndarray) -> float:
    print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
    if os.path.exists(BEST_MODEL_PATH):
        best_ckpt, _fc_w, _fc_b, new_dw = reload_weights_from_checkpoint(
            BEST_MODEL_PATH, runtime.weights, arch,
        )
        runtime.weights = new_dw
        print(f"Reloaded best checkpoint from epoch {int(best_ckpt['epoch'])} "
              f"with Val={float(best_ckpt['val_acc']):.2f}%")
    test_acc = evaluate(x_test, y_test, runtime.weights)
    print(f"Test Accuracy: {test_acc:.2f}%")
    return float(test_acc)


def main() -> dict[str, object]:
    arch = get_arch()
    x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_normalized_cifar10()
    runtime = init_cuda_runtime(arch)
    print_training_header(arch, x_train, x_val, x_test_final)
    fit = FitState()
    lr_state = LrState(LR_CONV1, LR_CONV, LR_FC)
    workspace = BatchWorkspace()
    train_rng = np.random.default_rng(TRAIN_SEED)

    try:
        try:
            for epoch in range(EPOCHS):
                with EpochTimer() as timer:
                    metrics = run_cuda_epoch(
                        runtime, workspace, arch, x_train, y_train, train_rng, lr_state
                    )
                    synchronize_if_available()
                    val_acc = evaluate(x_val, y_val, runtime.weights)
                    improved = fit.observe(epoch + 1, val_acc, MIN_DELTA)

                if improved:
                    save_best_checkpoint(epoch + 1, val_acc, lr_state, runtime, arch)
                    save_msg = " [saved best]"
                else:
                    save_msg = ""

                reduce_lr_if_due(fit, lr_state)
                print(format_epoch_summary(
                    epoch + 1, EPOCHS, metrics, val_acc, fit, lr_state, timer.elapsed, save_msg,
                    lr_separator=",",
                ))

                if fit.should_stop(EARLY_STOP_PATIENCE):
                    print(f"Early stopping after {epoch+1} epochs; "
                          f"best val {fit.best_val_acc:.2f}% at epoch {fit.best_epoch}.")
                    break

        finally:
            workspace.free()

        test_acc = run_final_evaluation(runtime, arch, x_test_final, y_test_final)
        print("\nDone!")
        return {
            'test_acc': test_acc,
            'best_model_path': BEST_MODEL_PATH,
        }
    finally:
        free_weights(runtime.velocities)
        runtime.velocities = None
        if runtime.adam is not None:
            free_weights(runtime.adam)
            runtime.adam = None
        free_weights(runtime.weights)
        runtime.weights = None


if __name__ == "__main__":
    main()
