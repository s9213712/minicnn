from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from minicnn.config.settings import (
    BATCH,
    EPOCHS,
    INIT_SEED,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    MOMENTUM,
    OPTIMIZER_TYPE,
    WEIGHT_DECAY,
)
from minicnn.models.initialization import init_weights
from minicnn.paths import ARTIFACTS_ROOT, BEST_MODELS_ROOT
from minicnn.training.checkpoints import (
    free_weights,
    init_adam_buffers,
    init_velocity_buffers,
    reload_weights_from_checkpoint,
    save_checkpoint,
    upload_weights,
)
from minicnn.training.cuda_batch import CudaRuntimeState
from minicnn.training.evaluation import evaluate

if TYPE_CHECKING:
    from minicnn.training.cuda_arch import CudaNetGeometry
    from minicnn.training.loop import LrState


def resolve_legacy_artifacts(best_model_filename: str) -> tuple[Path, str]:
    run_dir = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
    run_dir.mkdir(parents=True, exist_ok=True)
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    best_model_path = str(BEST_MODELS_ROOT / f"{run_dir.name}_{best_model_filename}")
    return run_dir, best_model_path


def init_cuda_runtime(arch: "CudaNetGeometry") -> CudaRuntimeState:
    *conv_arrays, fc_w, fc_b = init_weights(INIT_SEED, arch)
    use_adam = OPTIMIZER_TYPE.lower() == 'adam'
    ln_stages = [s for s in arch.conv_stages if s.layer_norm]
    ln_gamma = [np.ones(s.out_c, dtype=np.float32) for s in ln_stages]
    ln_beta = [np.zeros(s.out_c, dtype=np.float32) for s in ln_stages]
    bn_stages = [s for s in arch.conv_stages if s.batch_norm]
    bn_gamma = [np.ones(s.out_c, dtype=np.float32) for s in bn_stages]
    bn_beta = [np.zeros(s.out_c, dtype=np.float32) for s in bn_stages]
    bn_running_mean = [np.zeros(s.out_c, dtype=np.float32) for s in bn_stages]
    bn_running_var = [np.ones(s.out_c, dtype=np.float32) for s in bn_stages]
    return CudaRuntimeState(
        weights=upload_weights(
            conv_arrays, fc_w, fc_b,
            ln_gamma, ln_beta,
            bn_gamma, bn_beta, bn_running_mean, bn_running_var,
        ),
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
    print(
        f"LR_conv1={LR_CONV1}, LR_conv={LR_CONV}, LR_fc={LR_FC}, "
        f"momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, BATCH={BATCH}, EPOCHS={EPOCHS}"
    )
    print()


def save_best_checkpoint(
    best_model_path: str,
    epoch: int,
    val_acc: float,
    lr_state: "LrState",
    runtime: CudaRuntimeState,
    arch: "CudaNetGeometry",
) -> None:
    save_checkpoint(
        best_model_path, epoch, val_acc,
        lr_state.conv1, lr_state.conv, lr_state.fc,
        runtime.weights, arch,
    )


def run_final_evaluation(
    best_model_path: str,
    runtime: CudaRuntimeState,
    arch: "CudaNetGeometry",
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
    if os.path.exists(best_model_path):
        best_ckpt, _fc_w, _fc_b, new_dw = reload_weights_from_checkpoint(
            best_model_path, runtime.weights, arch,
        )
        runtime.weights = new_dw
        print(
            f"Reloaded best checkpoint from epoch {int(best_ckpt['epoch'])} "
            f"with Val={float(best_ckpt['val_acc']):.2f}%"
        )
    test_acc = evaluate(x_test, y_test, runtime.weights)
    print(f"Test Accuracy: {test_acc:.2f}%")
    return float(test_acc)


def cleanup_runtime(runtime: CudaRuntimeState) -> None:
    free_weights(runtime.velocities)
    runtime.velocities = None
    if runtime.adam is not None:
        free_weights(runtime.adam)
        runtime.adam = None
    free_weights(runtime.weights)
    runtime.weights = None
