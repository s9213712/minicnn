"""CUDA legacy batch-level training steps."""
from __future__ import annotations

from ctypes import c_float
from dataclasses import dataclass

import numpy as np

from minicnn.core.cuda_backend import (
    download_float_scalar,
    download_int_scalar,
    lib,
    update_conv,
)
from minicnn.config.settings import (
    CONV_GRAD_SPATIAL_NORMALIZE,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
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


@dataclass
class CudaRuntimeState:
    weights: DeviceWeights
    velocities: VelocityBuffers


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


def backward_fc_update(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    fc_input,
    batch_size: int,
    lr_state: LrState,
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
    lib.conv_update_fused(
        runtime.weights.fc_w,
        workspace.d_fc_grad_w,
        runtime.velocities.fc_w_vel,
        c_float(lr_state.fc),
        c_float(MOMENTUM),
        c_float(WEIGHT_DECAY),
        c_float(GRAD_CLIP_FC),
        c_float(1.0),
        arch.fc_out * arch.fc_in,
    )
    lib.conv_update_fused(
        runtime.weights.fc_b,
        workspace.d_fc_grad_b,
        runtime.velocities.fc_b_vel,
        c_float(lr_state.fc),
        c_float(MOMENTUM),
        c_float(0.0),
        c_float(GRAD_CLIP_BIAS),
        c_float(1.0),
        arch.fc_out,
    )


def backward_convs_update(
    runtime: CudaRuntimeState,
    workspace: BatchWorkspace,
    arch: CudaNetGeometry,
    batch_size: int,
    lr_state: LrState,
    log_grad: bool,
) -> None:
    lib.clip_inplace(workspace.d_pre_fc_grad_nchw, c_float(GRAD_POOL_CLIP), batch_size * arch.fc_in)
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
            c_float(LEAKY_ALPHA),
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

        lr = lr_state.conv1 if i == 0 else lr_state.conv
        spatial_norm = stage.h_out * stage.w_out if CONV_GRAD_SPATIAL_NORMALIZE else 1.0
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

        grad_nchw = workspace.d_input_nchw_grad[i]


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
    backward_fc_update(runtime, workspace, arch, fc_input, batch_size, lr_state)
    backward_convs_update(runtime, workspace, arch, batch_size, lr_state, log_grad)
