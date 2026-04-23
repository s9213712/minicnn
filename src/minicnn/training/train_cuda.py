#!/usr/bin/env python3
"""Train a VGG-style CUDA CNN on CIFAR-10.

Architecture is read from the YAML `model.conv_layers` list; adding or
removing layers requires only a YAML edit — no Python changes.
"""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from minicnn.training.cuda_arch import CudaNetGeometry

from minicnn.training.evaluation import evaluate
from minicnn.training.cuda_batch import CudaRuntimeState, synchronize_if_available, train_cuda_batch
from minicnn.training.cuda_epoch import augment_batch
from minicnn.training._legacy_cuda_runtime import (
    cleanup_runtime,
    init_cuda_runtime,
    print_training_header,
    resolve_legacy_artifacts,
    run_final_evaluation,
    save_best_checkpoint,
)
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
from minicnn.training.events import emit_training_event
from minicnn.config.settings import (
    BATCH,
    BEST_MODEL_FILENAME,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GRAD_DEBUG,
    GRAD_DEBUG_BATCHES,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    LR_PLATEAU_PATIENCE,
    LR_REDUCE_FACTOR,
    MIN_DELTA,
    MIN_LR,
    RANDOM_CROP_PADDING,
    HORIZONTAL_FLIP,
    TRAIN_SEED,
    get_arch,
)


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
        prev_loss_sum = metrics.loss_sum
        prev_correct = metrics.correct
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
            batch_loss = (metrics.loss_sum - prev_loss_sum) / max(n, 1)
            batch_acc = (metrics.correct - prev_correct) / max(n, 1) * 100.0
            emit_training_event(
                'batch_progress',
                {
                    'batch_idx': batch_idx + 1,
                    'num_batches': nbatches,
                    'loss': batch_loss,
                    'acc_percent': batch_acc,
                },
            )

    return metrics


def reduce_lr_if_due(fit: FitState, lr_state: LrState) -> None:
    if reduce_lr_on_plateau(fit, lr_state, LR_PLATEAU_PATIENCE, LR_REDUCE_FACTOR, MIN_LR):
        emit_training_event(
            'lr_reduced',
            {
                'conv1': lr_state.conv1,
                'conv': lr_state.conv,
                'fc': lr_state.fc,
                'label': 'LR',
            },
        )


def main() -> dict[str, object]:
    arch = get_arch()
    x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_normalized_cifar10()
    _run_dir, best_model_path = resolve_legacy_artifacts(BEST_MODEL_FILENAME)
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
                    save_best_checkpoint(best_model_path, epoch + 1, val_acc, lr_state, runtime, arch)
                    save_msg = " [saved best]"
                else:
                    save_msg = ""

                reduce_lr_if_due(fit, lr_state)
                print(format_epoch_summary(
                    epoch + 1, EPOCHS, metrics, val_acc, fit, lr_state, timer.elapsed, save_msg,
                    lr_separator=",",
                ))

                if fit.should_stop(EARLY_STOP_PATIENCE):
                    emit_training_event(
                        'legacy_early_stop',
                        {
                            'epoch': epoch + 1,
                            'best_val_acc': fit.best_val_acc,
                            'best_epoch': fit.best_epoch,
                        },
                    )
                    break

        finally:
            workspace.free()

        test_acc = run_final_evaluation(best_model_path, runtime, arch, x_test_final, y_test_final)
        print("\nDone!")
        return {
            'test_acc': test_acc,
            'best_model_path': best_model_path,
        }
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
