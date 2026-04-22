"""CUDA legacy batch-level training steps."""
from __future__ import annotations

from ctypes import c_float
from dataclasses import dataclass
import math

import numpy as np

from minicnn.core.cuda_backend import (
    g2h,
    is_lib_loaded,
    lib,
    update_adam,
    update_conv,
)
from minicnn.config.settings import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    CONV_GRAD_SPATIAL_NORMALIZE,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
    GRAD_CLIP_GLOBAL,
    MOMENTUM,
    WEIGHT_DECAY,
)
from minicnn.training.checkpoints import AdamBuffers, DeviceWeights, VelocityBuffers
from minicnn.training._cuda_batch_steps import (
    backward_convs_impl,
    backward_fc_impl,
    compute_loss_and_metrics_impl,
    final_activation_ptr as _final_activation_ptr,
    forward_convs_impl,
    forward_fc_impl,
)
from minicnn.training.cuda_arch import CudaNetGeometry
from minicnn.training.cuda_ops import (
    upload_int_to,
    upload_to,
)
from minicnn.training.cuda_workspace import BatchWorkspace
from minicnn.training.loop import LrState, RunningMetrics

# Precompute ctypes wrappers for scalar hyperparameters that never change at runtime.
_C_MOMENTUM      = c_float(MOMENTUM)
_C_WEIGHT_DECAY  = c_float(WEIGHT_DECAY)
_C_GRAD_CLIP_FC  = c_float(GRAD_CLIP_FC)
_C_GRAD_CLIP_BIAS = c_float(GRAD_CLIP_BIAS)
_C_ZERO          = c_float(0.0)


@dataclass
class CudaRuntimeState:
    weights: DeviceWeights
    velocities: VelocityBuffers
    adam: AdamBuffers | None = None
    adam_step: int = 0          # incremented each optimizer step


def global_clip_scale(total_grad_sq: float, max_norm: float, eps: float = 1e-12) -> float:
    """Return the gradient scale for global-norm clipping.

    A non-positive max_norm disables global clipping. The returned value is in
    (0, 1], so existing CUDA update kernels can apply it by increasing their
    gradient normalizer.
    """
    if max_norm <= 0.0:
        return 1.0
    norm = math.sqrt(max(float(total_grad_sq), 0.0))
    if norm <= max_norm:
        return 1.0
    return float(max_norm / (norm + eps))


def scaled_normalizer(base_normalizer: float, scale: float) -> float:
    if scale <= 0.0:
        return float('inf')
    return float(base_normalizer) / float(scale)


def device_grad_sq(d_grad, size: int, normalizer: float = 1.0) -> float:
    # NOTE: copies gradient buffer from device to host — incurs PCIe round-trip.
    # Enabling GRAD_CLIP_GLOBAL will therefore reduce throughput proportionally
    # to the number of parameter tensors in the network.
    if size <= 0 or not is_lib_loaded():
        return 0.0
    grad = g2h(d_grad, size).astype(np.float64, copy=False).reshape(-1)
    if normalizer != 1.0:
        grad = grad / float(normalizer)
    return float(np.dot(grad, grad))


def synchronize_if_available() -> None:
    if hasattr(lib, 'gpu_synchronize'):
        lib.gpu_synchronize()


def upload_batch(workspace: BatchWorkspace, x: np.ndarray, y: np.ndarray) -> None:
    upload_to(workspace.d_x, x)
    upload_int_to(workspace.d_y, y)


def final_activation_ptr(arch: CudaNetGeometry, workspace: BatchWorkspace) -> object:
    return _final_activation_ptr(arch, workspace)


def forward_convs(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    batch_size: int,
) -> object:
    return forward_convs_impl(runtime, workspace, arch, batch_size)


def forward_fc(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    fc_input,
    batch_size: int,
) -> None:
    forward_fc_impl(runtime, workspace, arch, fc_input, batch_size)


def compute_loss_and_metrics(
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    metrics: RunningMetrics,
    batch_size: int,
) -> None:
    compute_loss_and_metrics_impl(workspace, arch, metrics, batch_size)


def backward_fc(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    fc_input,
    batch_size: int,
) -> None:
    backward_fc_impl(runtime, workspace, arch, fc_input, batch_size)


def _adam_bias_corrections(step: int) -> tuple[float, float]:
    """Return (bias_corr1, bias_corr2) = (1-beta1^t, 1-beta2^t) for step t."""
    return (1.0 - ADAM_BETA1 ** step), (1.0 - ADAM_BETA2 ** step)


def update_fc(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    lr_state: LrState,
    grad_scale: float = 1.0,
) -> None:
    normalizer = scaled_normalizer(1.0, grad_scale)
    if runtime.adam is not None:
        bc1, bc2 = _adam_bias_corrections(runtime.adam_step)
        update_adam(
            runtime.weights.fc_w, workspace.d_fc_grad_w,
            runtime.adam.fc_w_m, runtime.adam.fc_w_v,
            lr_state.fc, ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
            WEIGHT_DECAY, GRAD_CLIP_FC,
            arch.fc_out * arch.fc_in, "fc_w",
            grad_normalizer=normalizer, bias_corr1=bc1, bias_corr2=bc2,
        )
        update_adam(
            runtime.weights.fc_b, workspace.d_fc_grad_b,
            runtime.adam.fc_b_m, runtime.adam.fc_b_v,
            lr_state.fc, ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
            0.0, GRAD_CLIP_BIAS,
            arch.fc_out, "fc_b",
            grad_normalizer=normalizer, bias_corr1=bc1, bias_corr2=bc2,
        )
    else:
        norm_c = c_float(normalizer)
        lib.conv_update_fused(
            runtime.weights.fc_w, workspace.d_fc_grad_w, runtime.velocities.fc_w_vel,
            c_float(lr_state.fc), _C_MOMENTUM, _C_WEIGHT_DECAY, _C_GRAD_CLIP_FC,
            norm_c, arch.fc_out * arch.fc_in,
        )
        lib.conv_update_fused(
            runtime.weights.fc_b, workspace.d_fc_grad_b, runtime.velocities.fc_b_vel,
            c_float(lr_state.fc), _C_MOMENTUM, _C_ZERO, _C_GRAD_CLIP_BIAS,
            norm_c, arch.fc_out,
        )


def conv_grad_normalizers(arch: CudaNetGeometry) -> list[float]:
    normalizers = []
    for stage in arch.conv_stages:
        normalizers.append(float(stage.h_out * stage.w_out if CONV_GRAD_SPATIAL_NORMALIZE else 1.0))
    return normalizers


def backward_convs(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    batch_size: int,
) -> None:
    backward_convs_impl(runtime, workspace, arch, batch_size)


def cuda_global_grad_scale(
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    max_norm: float,
    normalizers: list[float],
) -> float:
    if max_norm <= 0.0:
        return 1.0
    total_sq = device_grad_sq(workspace.d_fc_grad_w, arch.fc_out * arch.fc_in)
    total_sq += device_grad_sq(workspace.d_fc_grad_b, arch.fc_out)
    for i, stage in enumerate(arch.conv_stages):
        total_sq += device_grad_sq(workspace.d_w_grad[i], stage.weight_numel, normalizers[i])
    return global_clip_scale(total_sq, max_norm)


def update_convs(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    lr_state: LrState,
    grad_scale: float,
    log_grad: bool,
    normalizers: list[float],
) -> None:
    if runtime.adam is not None:
        bc1, bc2 = _adam_bias_corrections(runtime.adam_step)
        for i, stage in enumerate(arch.conv_stages):
            lr = lr_state.conv1 if i == 0 else lr_state.conv
            spatial_norm = scaled_normalizer(normalizers[i], grad_scale)
            update_adam(
                runtime.weights.conv_weights[i], workspace.d_w_grad[i],
                runtime.adam.conv_m[i], runtime.adam.conv_v[i],
                lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
                WEIGHT_DECAY, GRAD_CLIP_CONV,
                stage.weight_numel, f"conv{i + 1}",
                grad_normalizer=spatial_norm, bias_corr1=bc1, bias_corr2=bc2,
                log_grad=log_grad,
            )
            if stage.batch_norm:
                bn_idx = arch.bn_param_idx(i)
                update_adam(
                    runtime.weights.bn_gamma[bn_idx], workspace.d_bn_dgamma[i],
                    runtime.adam.bn_gamma_m[bn_idx], runtime.adam.bn_gamma_v[bn_idx],
                    lr_state.conv, ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
                    0.0, GRAD_CLIP_CONV,
                    stage.out_c, f"bn_gamma{i + 1}",
                    bias_corr1=bc1, bias_corr2=bc2,
                )
                update_adam(
                    runtime.weights.bn_beta[bn_idx], workspace.d_bn_dbeta[i],
                    runtime.adam.bn_beta_m[bn_idx], runtime.adam.bn_beta_v[bn_idx],
                    lr_state.conv, ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
                    0.0, GRAD_CLIP_BIAS,
                    stage.out_c, f"bn_beta{i + 1}",
                    bias_corr1=bc1, bias_corr2=bc2,
                )
    else:
        for i, stage in enumerate(arch.conv_stages):
            lr = lr_state.conv1 if i == 0 else lr_state.conv
            spatial_norm = scaled_normalizer(normalizers[i], grad_scale)
            update_conv(
                runtime.weights.conv_weights[i],
                workspace.d_w_grad[i],
                runtime.velocities.conv_velocities[i],
                lr,
                MOMENTUM,
                stage.weight_numel,
                f"conv{i + 1}",
                WEIGHT_DECAY,
                GRAD_CLIP_CONV,
                spatial_norm,
                log_grad,
            )
            if stage.batch_norm:
                bn_idx = arch.bn_param_idx(i)
                norm_c = c_float(scaled_normalizer(1.0, grad_scale))
                lib.conv_update_fused(
                    runtime.weights.bn_gamma[bn_idx], workspace.d_bn_dgamma[i],
                    runtime.velocities.bn_gamma_vel[bn_idx],
                    c_float(lr_state.conv), _C_MOMENTUM, _C_ZERO, c_float(GRAD_CLIP_CONV),
                    norm_c, stage.out_c,
                )
                lib.conv_update_fused(
                    runtime.weights.bn_beta[bn_idx], workspace.d_bn_dbeta[i],
                    runtime.velocities.bn_beta_vel[bn_idx],
                    c_float(lr_state.conv), _C_MOMENTUM, _C_ZERO, _C_GRAD_CLIP_BIAS,
                    norm_c, stage.out_c,
                )


def train_cuda_batch(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    x: np.ndarray,
    y: np.ndarray,
    lr_state: LrState,
    metrics: RunningMetrics,
    log_grad: bool,
) -> None:
    batch_size = int(x.shape[0])
    upload_batch(workspace, x, y)
    fc_input = forward_convs(runtime, workspace, arch, batch_size)
    forward_fc(runtime, workspace, arch, fc_input, batch_size)
    compute_loss_and_metrics(workspace, arch, metrics, batch_size)
    backward_fc(runtime, workspace, arch, fc_input, batch_size)
    backward_convs(runtime, workspace, arch, batch_size)
    normalizers = conv_grad_normalizers(arch)
    grad_scale = cuda_global_grad_scale(workspace, arch, GRAD_CLIP_GLOBAL, normalizers)
    if runtime.adam is not None:
        runtime.adam_step += 1
    update_fc(runtime, workspace, arch, lr_state, grad_scale)
    update_convs(runtime, workspace, arch, lr_state, grad_scale, log_grad, normalizers)
