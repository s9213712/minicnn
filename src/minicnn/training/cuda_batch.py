"""CUDA legacy batch-level training steps."""
from __future__ import annotations

from ctypes import c_float
from dataclasses import dataclass
import math

import numpy as np

from minicnn.core.cuda_backend import (
    download_float_scalar,
    download_int_scalar,
    g2h,
    is_lib_loaded,
    lib,
    update_conv,
)
from minicnn.config.settings import (
    CONV_GRAD_SPATIAL_NORMALIZE,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
    GRAD_CLIP_GLOBAL,
    GRAD_POOL_CLIP,
    LEAKY_ALPHA,
    MOMENTUM,
    WEIGHT_DECAY,
)
from minicnn.training.checkpoints import DeviceWeights, VelocityBuffers
from minicnn.training.cuda_arch import CudaNetGeometry
from minicnn.training.cuda_ops import (
    cnhw_to_nchw_into,
    conv_forward_into,
    maxpool_forward_into,
    nchw_to_cnhw_into,
    upload_int_to,
    upload_to,
    zero_floats,
)
from minicnn.training.cuda_workspace import BatchWorkspace
from minicnn.training.loop import LrState, RunningMetrics

# Precompute ctypes wrappers for scalar hyperparameters that never change at runtime.
_C_MOMENTUM      = c_float(MOMENTUM)
_C_WEIGHT_DECAY  = c_float(WEIGHT_DECAY)
_C_GRAD_CLIP_FC  = c_float(GRAD_CLIP_FC)
_C_GRAD_CLIP_BIAS = c_float(GRAD_CLIP_BIAS)
_C_GRAD_POOL_CLIP = c_float(GRAD_POOL_CLIP)
_C_LEAKY_ALPHA   = c_float(LEAKY_ALPHA)
_C_ZERO          = c_float(0.0)


@dataclass
class CudaRuntimeState:
    weights: DeviceWeights
    velocities: VelocityBuffers


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


def _prev_nchw(arch: CudaNetGeometry, i: int, workspace: BatchWorkspace) -> object:
    if i == 0:
        return workspace.d_x
    prev_stage = arch.conv_stages[i - 1]
    return workspace.d_pool_nchw[i - 1] if prev_stage.pool else workspace.d_conv_nchw[i - 1]


def final_activation_ptr(arch: CudaNetGeometry, workspace: BatchWorkspace) -> object:
    last_stage = arch.conv_stages[-1]
    return workspace.d_pool_nchw[-1] if last_stage.pool else workspace.d_conv_nchw[-1]


def forward_convs(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    batch_size: int,
) -> object:
    for i, stage in enumerate(arch.conv_stages):
        conv_forward_into(
            _prev_nchw(arch, i, workspace),
            runtime.weights.conv_weights[i],
            workspace.d_col[i],
            workspace.d_conv_raw[i],
            batch_size,
            stage.in_c,
            stage.h_in,
            stage.w_in,
            stage.out_c,
        )
        if stage.pool:
            maxpool_forward_into(
                workspace.d_conv_raw[i],
                workspace.d_pool[i],
                workspace.d_max_idx[i],
                batch_size,
                stage.out_c,
                stage.h_out,
                stage.w_out,
            )
            cnhw_to_nchw_into(
                workspace.d_pool[i],
                workspace.d_pool_nchw[i],
                batch_size,
                stage.out_c,
                stage.ph,
                stage.pw,
            )
        else:
            cnhw_to_nchw_into(
                workspace.d_conv_raw[i],
                workspace.d_conv_nchw[i],
                batch_size,
                stage.out_c,
                stage.h_out,
                stage.w_out,
            )

    return final_activation_ptr(arch, workspace)


def forward_fc(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    fc_input,
    batch_size: int,
) -> None:
    lib.dense_forward(
        fc_input,
        runtime.weights.fc_w,
        runtime.weights.fc_b,
        workspace.d_fc_out,
        batch_size,
        arch.fc_in,
        arch.fc_out,
    )


def compute_loss_and_metrics(
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    metrics: RunningMetrics,
    batch_size: int,
) -> None:
    lib.gpu_memset(workspace.d_loss_sum, 0, 4)
    lib.gpu_memset(workspace.d_correct, 0, 4)
    lib.softmax_xent_grad_loss_acc(
        workspace.d_fc_out,
        workspace.d_y,
        workspace.d_probs,
        workspace.d_grad_logits,
        workspace.d_loss_sum,
        workspace.d_correct,
        batch_size,
        arch.fc_out,
    )
    metrics.update(
        download_float_scalar(workspace.d_loss_sum),
        download_int_scalar(workspace.d_correct),
        batch_size,
    )


def backward_fc(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    fc_input,
    batch_size: int,
) -> None:
    lib.dense_backward_full(
        workspace.d_grad_logits,
        fc_input,
        runtime.weights.fc_w,
        workspace.d_pre_fc_grad_nchw,
        workspace.d_fc_grad_w,
        workspace.d_fc_grad_b,
        batch_size,
        arch.fc_in,
        arch.fc_out,
    )


def update_fc(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    lr_state: LrState,
    grad_scale: float = 1.0,
) -> None:
    normalizer = c_float(scaled_normalizer(1.0, grad_scale))
    lib.conv_update_fused(
        runtime.weights.fc_w,
        workspace.d_fc_grad_w,
        runtime.velocities.fc_w_vel,
        c_float(lr_state.fc),
        _C_MOMENTUM,
        _C_WEIGHT_DECAY,
        _C_GRAD_CLIP_FC,
        normalizer,
        arch.fc_out * arch.fc_in,
    )
    lib.conv_update_fused(
        runtime.weights.fc_b,
        workspace.d_fc_grad_b,
        runtime.velocities.fc_b_vel,
        c_float(lr_state.fc),
        _C_MOMENTUM,
        _C_ZERO,
        _C_GRAD_CLIP_BIAS,
        normalizer,
        arch.fc_out,
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
    lib.clip_inplace(workspace.d_pre_fc_grad_nchw, _C_GRAD_POOL_CLIP, batch_size * arch.fc_in)
    grad_nchw = workspace.d_pre_fc_grad_nchw

    for i in reversed(range(arch.n_conv)):
        stage = arch.conv_stages[i]

        if stage.pool:
            nchw_to_cnhw_into(
                grad_nchw,
                workspace.d_pool_grad_cnhw[i],
                batch_size,
                stage.out_c,
                stage.ph,
                stage.pw,
            )
            zero_floats(workspace.d_conv_raw_grad[i], stage.out_c * batch_size * stage.h_out * stage.w_out)
            lib.maxpool_backward_use_idx(
                workspace.d_pool_grad_cnhw[i],
                workspace.d_max_idx[i],
                workspace.d_conv_raw_grad[i],
                batch_size,
                stage.out_c,
                stage.h_out,
                stage.w_out,
            )
        else:
            nchw_to_cnhw_into(
                grad_nchw,
                workspace.d_conv_raw_grad[i],
                batch_size,
                stage.out_c,
                stage.h_out,
                stage.w_out,
            )

        lib.leaky_relu_backward(
            workspace.d_conv_raw[i],
            workspace.d_conv_raw_grad[i],
            _C_LEAKY_ALPHA,
            stage.out_c * batch_size * stage.h_out * stage.w_out,
        )
        lib.conv_backward_precol(
            workspace.d_conv_raw_grad[i],
            _prev_nchw(arch, i, workspace),
            runtime.weights.conv_weights[i],
            workspace.d_w_grad[i],
            workspace.d_input_nchw_grad[i],
            workspace.d_col[i],
            batch_size,
            stage.in_c,
            stage.h_in,
            stage.w_in,
            stage.kh,
            stage.kw,
            stage.h_out,
            stage.w_out,
            stage.out_c,
        )

        grad_nchw = workspace.d_input_nchw_grad[i]


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
    update_fc(runtime, workspace, arch, lr_state, grad_scale)
    update_convs(runtime, workspace, arch, lr_state, grad_scale, log_grad, normalizers)
