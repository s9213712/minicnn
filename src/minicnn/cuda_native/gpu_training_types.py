from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NativeGpuLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    grad_input: np.ndarray
    grad_weight: np.ndarray
    grad_bias: np.ndarray
    updated_weight: np.ndarray
    updated_bias: np.ndarray
    updated_weight_velocity: np.ndarray | None
    updated_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]
    updated_weight_m: np.ndarray | None = None
    updated_weight_v: np.ndarray | None = None
    updated_bias_m: np.ndarray | None = None
    updated_bias_v: np.ndarray | None = None
    updated_weight_rmsprop_v: np.ndarray | None = None
    updated_weight_rmsprop_buf: np.ndarray | None = None
    updated_bias_rmsprop_v: np.ndarray | None = None
    updated_bias_rmsprop_buf: np.ndarray | None = None


@dataclass(frozen=True)
class NativeGpuTwoLinearReluTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    grad_hidden: np.ndarray
    grad_input: np.ndarray
    grad_weight1: np.ndarray
    grad_bias1: np.ndarray
    grad_weight2: np.ndarray
    grad_bias2: np.ndarray
    updated_weight1: np.ndarray
    updated_bias1: np.ndarray
    updated_weight2: np.ndarray
    updated_bias2: np.ndarray
    updated_weight1_velocity: np.ndarray | None
    updated_bias1_velocity: np.ndarray | None
    updated_weight2_velocity: np.ndarray | None
    updated_bias2_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuMlpTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    grad_input: np.ndarray
    grad_params: dict[str, np.ndarray]
    updated_params: dict[str, np.ndarray]
    updated_velocity: dict[str, np.ndarray] | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuPoolLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    pooled: np.ndarray
    grad_pooled: np.ndarray
    grad_input: np.ndarray
    grad_weight: np.ndarray
    grad_bias: np.ndarray
    updated_weight: np.ndarray
    updated_bias: np.ndarray
    updated_weight_velocity: np.ndarray | None
    updated_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuBatchNormLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    bn_output: np.ndarray
    x_hat: np.ndarray
    batch_mean: np.ndarray
    batch_inv_std: np.ndarray
    grad_bn_output: np.ndarray
    grad_input: np.ndarray
    grad_bn_weight: np.ndarray
    grad_bn_bias: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_bn_weight: np.ndarray
    updated_bn_bias: np.ndarray
    updated_running_mean: np.ndarray
    updated_running_var: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_bn_weight_velocity: np.ndarray | None
    updated_bn_bias_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuLayerNorm2dLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    norm_output: np.ndarray
    grad_logits: np.ndarray
    grad_norm_output: np.ndarray
    grad_input: np.ndarray
    grad_norm_weight: np.ndarray
    grad_norm_bias: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_norm_weight: np.ndarray
    updated_norm_bias: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_norm_weight_velocity: np.ndarray | None
    updated_norm_bias_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuLayerNormLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    norm_output: np.ndarray
    grad_logits: np.ndarray
    grad_norm_output: np.ndarray
    grad_input: np.ndarray
    grad_norm_weight: np.ndarray
    grad_norm_bias: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_norm_weight: np.ndarray
    updated_norm_bias: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_norm_weight_velocity: np.ndarray | None
    updated_norm_bias_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuGroupNormLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    norm_output: np.ndarray
    grad_logits: np.ndarray
    grad_norm_output: np.ndarray
    grad_input: np.ndarray
    grad_norm_weight: np.ndarray
    grad_norm_bias: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_norm_weight: np.ndarray
    updated_norm_bias: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_norm_weight_velocity: np.ndarray | None
    updated_norm_bias_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuConvLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    conv_output: np.ndarray
    grad_logits: np.ndarray
    grad_conv_output: np.ndarray
    grad_input: np.ndarray
    grad_conv_weight: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_conv_weight: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_conv_weight_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]
    pooled_output: np.ndarray | None = None
    grad_pooled: np.ndarray | None = None


@dataclass(frozen=True)
class NativeGpuDepthwiseLayerNorm2dLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    conv_output: np.ndarray
    norm_output: np.ndarray
    grad_logits: np.ndarray
    grad_norm_output: np.ndarray
    grad_conv_output: np.ndarray
    grad_input: np.ndarray
    grad_conv_weight: np.ndarray
    grad_norm_weight: np.ndarray
    grad_norm_bias: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_conv_weight: np.ndarray
    updated_norm_weight: np.ndarray
    updated_norm_bias: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_conv_weight_velocity: np.ndarray | None
    updated_norm_weight_velocity: np.ndarray | None
    updated_norm_bias_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuDepthwiseLayerNorm2dPointwiseLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    depthwise_output: np.ndarray
    norm_output: np.ndarray
    pointwise_output: np.ndarray
    grad_logits: np.ndarray
    grad_pointwise_output: np.ndarray
    grad_norm_output: np.ndarray
    grad_depthwise_output: np.ndarray
    grad_input: np.ndarray
    grad_depthwise_weight: np.ndarray
    grad_norm_weight: np.ndarray
    grad_norm_bias: np.ndarray
    grad_pointwise_weight: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_depthwise_weight: np.ndarray
    updated_norm_weight: np.ndarray
    updated_norm_bias: np.ndarray
    updated_pointwise_weight: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_depthwise_weight_velocity: np.ndarray | None
    updated_norm_weight_velocity: np.ndarray | None
    updated_norm_bias_velocity: np.ndarray | None
    updated_pointwise_weight_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    depthwise_output: np.ndarray
    norm_output: np.ndarray
    pointwise1_output: np.ndarray
    activation_output: np.ndarray
    pointwise2_output: np.ndarray
    grad_logits: np.ndarray
    grad_pointwise2_output: np.ndarray
    grad_activation_output: np.ndarray
    grad_pointwise1_output: np.ndarray
    grad_norm_output: np.ndarray
    grad_depthwise_output: np.ndarray
    grad_input: np.ndarray
    grad_depthwise_weight: np.ndarray
    grad_norm_weight: np.ndarray
    grad_norm_bias: np.ndarray
    grad_pointwise1_weight: np.ndarray
    grad_pointwise2_weight: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_depthwise_weight: np.ndarray
    updated_norm_weight: np.ndarray
    updated_norm_bias: np.ndarray
    updated_pointwise1_weight: np.ndarray
    updated_pointwise2_weight: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_depthwise_weight_velocity: np.ndarray | None
    updated_norm_weight_velocity: np.ndarray | None
    updated_norm_bias_velocity: np.ndarray | None
    updated_pointwise1_weight_velocity: np.ndarray | None
    updated_pointwise2_weight_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuTwoConvReluPoolLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    conv1_output: np.ndarray
    conv2_output: np.ndarray
    pooled_output: np.ndarray
    grad_logits: np.ndarray
    grad_pooled: np.ndarray
    grad_conv2_output: np.ndarray
    grad_conv1_output: np.ndarray
    grad_input: np.ndarray
    grad_conv1_weight: np.ndarray
    grad_conv2_weight: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_conv1_weight: np.ndarray
    updated_conv2_weight: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_conv1_weight_velocity: np.ndarray | None
    updated_conv2_weight_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]
