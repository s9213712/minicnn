"""Compatibility layer exposing legacy module-level constants.

Legacy training code imports constants directly from this module. The CLI loads a YAML config,
then calls `apply_experiment_config()` before importing the trainer implementation.

Architecture constants are now derived from `CudaNetGeometry` in a loop, so changing
`model.conv_layers` in the YAML is all that is required to adjust the network depth.
Backward-compat aliases (C1_IN, C1_OUT, H1, W1, P1H …) are still generated for the
first four stages so that `train_torch_baseline.py` continues to work unchanged.
"""
from __future__ import annotations

import os
from typing import Any

from minicnn.config.parsing import parse_bool
from .schema import ExperimentConfig

_current_config: ExperimentConfig | None = None
_arch = None  # CudaNetGeometry, set by apply_experiment_config


def _parse_bool(value: str) -> bool:
    return parse_bool(value, label='environment boolean')


_ENV_OVERRIDES = {
    "BATCH": ("MINICNN_BATCH", int),
    "EPOCHS": ("MINICNN_EPOCHS", int),
    "EVAL_MAX_BATCHES": ("MINICNN_EVAL_MAX_BATCHES", int),
    "N_TRAIN": ("MINICNN_N_TRAIN", int),
    "N_VAL": ("MINICNN_N_VAL", int),
    "DATASET_SEED": ("MINICNN_DATASET_SEED", int),
    "INIT_SEED": ("MINICNN_INIT_SEED", int),
    "TRAIN_SEED": ("MINICNN_TRAIN_SEED", int),
    "RANDOM_CROP_PADDING": ("MINICNN_RANDOM_CROP_PADDING", int),
    "HORIZONTAL_FLIP": ("MINICNN_HORIZONTAL_FLIP", _parse_bool),
    "LR_CONV1": ("MINICNN_LR_CONV1", float),
    "LR_CONV": ("MINICNN_LR_CONV", float),
    "LR_FC": ("MINICNN_LR_FC", float),
    "MOMENTUM": ("MINICNN_MOMENTUM", float),
    "WEIGHT_DECAY": ("MINICNN_WEIGHT_DECAY", float),
    "GRAD_CLIP_GLOBAL": ("MINICNN_GRAD_CLIP_GLOBAL", float),
}


def _apply_env_overrides(values: dict[str, Any]) -> None:
    for key, (env_name, parser) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw not in {None, ""}:
            values[key] = parser(raw)


def get_arch():
    """Return the CudaNetGeometry built from the current config."""
    if _arch is None:
        raise RuntimeError("apply_experiment_config() has not been called yet")
    return _arch


def apply_experiment_config(cfg: ExperimentConfig) -> None:
    global _current_config, _arch
    _current_config = cfg

    from minicnn.training.cuda_arch import CudaNetGeometry
    arch = CudaNetGeometry.from_config(cfg.model)
    _arch = arch

    values: dict[str, Any] = {
        "BATCH": cfg.train.batch_size,
        "EPOCHS": cfg.train.epochs,
        "EVAL_MAX_BATCHES": cfg.train.eval_max_batches,
        "N_TRAIN": cfg.train.n_train,
        "N_VAL": cfg.train.n_val,
        "TRAIN_BATCH_IDS": tuple(cfg.train.train_batch_ids),
        "DATASET_SEED": cfg.train.dataset_seed,
        "INIT_SEED": cfg.train.init_seed,
        "TRAIN_SEED": cfg.train.train_seed,
        "RANDOM_CROP_PADDING": cfg.train.random_crop_padding,
        "HORIZONTAL_FLIP": cfg.train.horizontal_flip,
        "EARLY_STOP_PATIENCE": cfg.train.early_stop_patience,
        "MIN_DELTA": cfg.train.min_delta,
        "LR_CONV1": cfg.optim.lr_conv1,
        "LR_CONV": cfg.optim.lr_conv,
        "LR_FC": cfg.optim.lr_fc,
        "LR_PLATEAU_PATIENCE": cfg.optim.lr_plateau_patience,
        "LR_REDUCE_FACTOR": cfg.optim.lr_reduce_factor,
        "MIN_LR": cfg.optim.min_lr,
        "MOMENTUM": cfg.optim.momentum,
        "LEAKY_ALPHA": cfg.optim.leaky_alpha,
        "WEIGHT_DECAY": cfg.optim.weight_decay,
        "GRAD_CLIP_CONV": cfg.optim.grad_clip_conv,
        "GRAD_CLIP_FC": cfg.optim.grad_clip_fc,
        "GRAD_CLIP_BIAS": cfg.optim.grad_clip_bias,
        "GRAD_POOL_CLIP": cfg.optim.grad_pool_clip,
        "GRAD_CLIP_GLOBAL": cfg.optim.grad_clip_global,
        "CONV_GRAD_SPATIAL_NORMALIZE": cfg.optim.conv_grad_spatial_normalize,
        "GRAD_DEBUG": cfg.runtime.grad_debug,
        "GRAD_DEBUG_BATCHES": cfg.runtime.grad_debug_batches,
        "BEST_MODEL_FILENAME": cfg.runtime.best_model_filename,
        "H": cfg.model.h,
        "W": cfg.model.w,
        "KH": cfg.model.kh,
        "KW": cfg.model.kw,
        "FC_IN": arch.fc_in,
        "FC_OUT": arch.fc_out,
    }
    _apply_env_overrides(values)

    # Backward-compat per-stage aliases used by train_torch_baseline.py and others.
    # Named aliases are generated for up to 4 stages; stage counting is 1-based.
    _pool_num = 0
    for i, s in enumerate(arch.conv_stages):
        n = i + 1  # 1-based
        values[f"C{n}_IN"] = s.in_c
        values[f"C{n}_OUT"] = s.out_c
        values[f"H{n}"] = s.h_out
        values[f"W{n}"] = s.w_out
        if s.pool:
            _pool_num += 1
            values[f"P{_pool_num}H"] = s.ph
            values[f"P{_pool_num}W"] = s.pw
        if n >= 4:
            break  # only alias up to 4 stages for compat

    globals().update(values)


def current_config() -> ExperimentConfig | None:
    return _current_config


def summarize() -> dict[str, Any]:
    arch = get_arch()
    return {
        "train": {
            "batch_size": BATCH,
            "epochs": EPOCHS,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "eval_max_batches": EVAL_MAX_BATCHES,
            "dataset_seed": DATASET_SEED,
            "init_seed": INIT_SEED,
            "train_seed": TRAIN_SEED,
        },
        "optimizer": {
            "lr_conv1": LR_CONV1,
            "lr_conv": LR_CONV,
            "lr_fc": LR_FC,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_global": GRAD_CLIP_GLOBAL,
        },
        "model": {
            "stages": [
                {
                    "in_c": s.in_c, "out_c": s.out_c,
                    "h_out": s.h_out, "w_out": s.w_out,
                    "pool": s.pool, "ph": s.ph, "pw": s.pw,
                }
                for s in arch.conv_stages
            ],
            "input_hw": [H, W],
            "kernel_hw": [KH, KW],
            "fc_in": arch.fc_in,
        },
    }


apply_experiment_config(ExperimentConfig())
