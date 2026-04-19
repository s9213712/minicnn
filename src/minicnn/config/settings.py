"""Compatibility layer exposing legacy module-level constants.

Legacy training code imports constants directly from this module. The CLI loads a YAML config,
then calls `apply_experiment_config()` before importing the trainer implementation.
"""
from __future__ import annotations

import os
from typing import Any

from .schema import ExperimentConfig

_current_config: ExperimentConfig | None = None

_ENV_OVERRIDES = {
    "BATCH": ("MINICNN_BATCH", int),
    "EPOCHS": ("MINICNN_EPOCHS", int),
    "EVAL_MAX_BATCHES": ("MINICNN_EVAL_MAX_BATCHES", int),
    "N_TRAIN": ("MINICNN_N_TRAIN", int),
    "N_VAL": ("MINICNN_N_VAL", int),
    "DATASET_SEED": ("MINICNN_DATASET_SEED", int),
    "INIT_SEED": ("MINICNN_INIT_SEED", int),
    "TRAIN_SEED": ("MINICNN_TRAIN_SEED", int),
    "LR_CONV1": ("MINICNN_LR_CONV1", float),
    "LR_CONV": ("MINICNN_LR_CONV", float),
    "LR_FC": ("MINICNN_LR_FC", float),
    "MOMENTUM": ("MINICNN_MOMENTUM", float),
    "WEIGHT_DECAY": ("MINICNN_WEIGHT_DECAY", float),
}


def _apply_env_overrides(values: dict[str, Any]) -> None:
    for key, (env_name, parser) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw not in {None, ""}:
            values[key] = parser(raw)


def _compute_shape_fields(values: dict[str, Any]) -> dict[str, Any]:
    H, W = values["H"], values["W"]
    KH, KW = values["KH"], values["KW"]
    H1, W1 = H - KH + 1, W - KW + 1
    H2, W2 = H1 - KH + 1, W1 - KW + 1
    P1H, P1W = H2 // 2, W2 // 2
    H3, W3 = P1H - KH + 1, P1W - KW + 1
    H4, W4 = H3 - KH + 1, W3 - KW + 1
    P2H, P2W = H4 // 2, W4 // 2
    FC_IN = values["C4_OUT"] * P2H * P2W
    shapes = {
        "H1": H1, "W1": W1, "H2": H2, "W2": W2, "P1H": P1H, "P1W": P1W,
        "H3": H3, "W3": W3, "H4": H4, "W4": W4, "P2H": P2H, "P2W": P2W,
        "FC_IN": FC_IN,
    }
    invalid = {name: value for name, value in shapes.items() if value <= 0}
    if invalid:
        raise ValueError(f"Invalid legacy CUDA model geometry: {invalid}")
    if H2 % 2 != 0 or W2 % 2 != 0 or H4 % 2 != 0 or W4 % 2 != 0:
        raise ValueError("Legacy CUDA model geometry requires even H2/W2 and H4/W4 before pooling")
    return {
        "H1": H1, "W1": W1, "H2": H2, "W2": W2, "P1H": P1H, "P1W": P1W,
        "H3": H3, "W3": W3, "H4": H4, "W4": W4, "P2H": P2H, "P2W": P2W,
        "FC_IN": FC_IN,
    }


def _validate_channel_links(values: dict[str, Any]) -> None:
    expected = {
        "C2_IN": values["C1_OUT"],
        "C3_IN": values["C2_OUT"],
        "C4_IN": values["C3_OUT"],
    }
    mismatches = {
        name: (values[name], expected_value)
        for name, expected_value in expected.items()
        if values[name] != expected_value
    }
    if mismatches:
        raise ValueError(f"Invalid legacy CUDA channel links: {mismatches}")


def apply_experiment_config(cfg: ExperimentConfig) -> None:
    global _current_config
    _current_config = cfg
    values = {
        "BATCH": cfg.train.batch_size,
        "EPOCHS": cfg.train.epochs,
        "EVAL_MAX_BATCHES": cfg.train.eval_max_batches,
        "N_TRAIN": cfg.train.n_train,
        "N_VAL": cfg.train.n_val,
        "TRAIN_BATCH_IDS": tuple(cfg.train.train_batch_ids),
        "DATASET_SEED": cfg.train.dataset_seed,
        "INIT_SEED": cfg.train.init_seed,
        "TRAIN_SEED": cfg.train.train_seed,
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
        "CONV_GRAD_SPATIAL_NORMALIZE": cfg.optim.conv_grad_spatial_normalize,
        "GRAD_DEBUG": cfg.runtime.grad_debug,
        "GRAD_DEBUG_BATCHES": cfg.runtime.grad_debug_batches,
        "BEST_MODEL_FILENAME": cfg.runtime.best_model_filename,
        "C1_IN": cfg.model.c1_in,
        "C1_OUT": cfg.model.c1_out,
        "C2_IN": cfg.model.c2_in,
        "C2_OUT": cfg.model.c2_out,
        "C3_IN": cfg.model.c3_in,
        "C3_OUT": cfg.model.c3_out,
        "C4_IN": cfg.model.c4_in,
        "C4_OUT": cfg.model.c4_out,
        "H": cfg.model.h,
        "W": cfg.model.w,
        "KH": cfg.model.kh,
        "KW": cfg.model.kw,
    }
    _apply_env_overrides(values)
    _validate_channel_links(values)
    values.update(_compute_shape_fields(values))
    globals().update(values)


def current_config() -> ExperimentConfig | None:
    return _current_config


apply_experiment_config(ExperimentConfig())
