from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.graph import NativeGraph


@dataclass
class NativeTrainingContext:
    cfg: dict[str, Any]
    graph: NativeGraph
    params: dict[str, np.ndarray]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    input_shape: tuple[int, ...]
    planner_summary: dict[str, Any]
    batch_size: int
    epochs: int
    lr: float
    optimizer_type: str
    weight_decay: float
    momentum: float
    grad_clip_global: float
    grad_accum_steps: int
    amp: bool
    amp_loss_scale: float
    amp_dynamic_scale: bool
    amp_scale_growth: float
    amp_scale_backoff: float
    amp_scale_window: int
    loss_type: str
    model_cfg: dict[str, Any]
    loss_cfg: dict[str, Any]
    optimizer_cfg: dict[str, Any]
    scheduler_cfg: dict[str, Any]
    support_tier_assessment: dict[str, Any]
    execution_mode: str
    selected_execution_mode: str
    tensor_execution_device: str
    execution_mode_policy: dict[str, Any]
    device_runtime: DeviceRuntime
