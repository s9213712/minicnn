"""Forward/backward CUDA legacy batch helpers."""

from __future__ import annotations

from ctypes import c_float
from typing import TYPE_CHECKING

from minicnn.config.settings import CUDA_LOSS_TYPE, GRAD_POOL_CLIP, LEAKY_ALPHA
from minicnn.core.cuda_backend import download_float_scalar, download_int_scalar, lib
from minicnn.training.cuda_ops import (
    cnhw_to_nchw_into,
    conv_forward_into,
    maxpool_forward_into,
    nchw_to_cnhw_into,
    zero_floats,
)

if TYPE_CHECKING:
    from minicnn.training.cuda_batch import CudaRuntimeState
    from minicnn.training.cuda_arch import CudaNetGeometry
    from minicnn.training.cuda_workspace import BatchWorkspace
    from minicnn.training.loop import RunningMetrics


_C_GRAD_POOL_CLIP = c_float(GRAD_POOL_CLIP)
_C_LEAKY_ALPHA = c_float(LEAKY_ALPHA)


def _prev_nchw(arch: "CudaNetGeometry", i: int, workspace: "BatchWorkspace") -> object:
    if i == 0:
        return workspace.d_x
    prev = arch.conv_stages[i - 1]
    if prev.pool:
        return workspace.d_pool_nchw[i - 1]
    if prev.batch_norm:
        return workspace.d_bn_out[i - 1]
    if prev.layer_norm:
        return workspace.d_ln_out[i - 1]
    return workspace.d_conv_nchw[i - 1]


def final_activation_ptr(arch: "CudaNetGeometry", workspace: "BatchWorkspace") -> object:
    last = arch.conv_stages[-1]
    if last.pool:
        return workspace.d_pool_nchw[-1]
    if last.batch_norm:
        return workspace.d_bn_out[-1]
    if last.layer_norm:
        return workspace.d_ln_out[-1]
    return workspace.d_conv_nchw[-1]


def forward_convs_impl(
    runtime: "CudaRuntimeState",
    workspace: "BatchWorkspace",
    arch: "CudaNetGeometry",
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
            if stage.layer_norm:
                ln_idx = arch.ln_param_idx(i)
                lib.layer_norm_forward(
                    workspace.d_ln_out[i],
                    workspace.d_conv_nchw[i],
                    runtime.weights.ln_gamma[ln_idx],
                    runtime.weights.ln_beta[ln_idx],
                    batch_size, stage.out_c, stage.h_out, stage.w_out,
                    c_float(1e-5),
                )
            elif stage.batch_norm:
                bn_idx = arch.bn_param_idx(i)
                lib.bn_train_forward(
                    workspace.d_bn_out[i],
                    workspace.d_conv_nchw[i],
                    workspace.d_bn_x_hat[i],
                    workspace.d_bn_mean[i],
                    workspace.d_bn_inv_std[i],
                    runtime.weights.bn_running_mean[bn_idx],
                    runtime.weights.bn_running_var[bn_idx],
                    runtime.weights.bn_gamma[bn_idx],
                    runtime.weights.bn_beta[bn_idx],
                    batch_size, stage.out_c, stage.h_out, stage.w_out,
                    c_float(1e-5), c_float(0.1),
                )

    return final_activation_ptr(arch, workspace)


def forward_fc_impl(
    runtime: "CudaRuntimeState",
    workspace: "BatchWorkspace",
    arch: "CudaNetGeometry",
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


def compute_loss_and_metrics_impl(
    workspace: "BatchWorkspace",
    arch: "CudaNetGeometry",
    metrics: "RunningMetrics",
    batch_size: int,
) -> None:
    lib.gpu_memset(workspace.d_loss_sum, 0, 4)
    lib.gpu_memset(workspace.d_correct, 0, 4)
    loss_type = CUDA_LOSS_TYPE
    if loss_type == 'mse':
        lib.mse_fwd_grad_loss_acc(
            workspace.d_fc_out,
            workspace.d_y,
            workspace.d_grad_logits,
            workspace.d_loss_sum,
            workspace.d_correct,
            batch_size,
            arch.fc_out,
        )
    elif loss_type == 'bce':
        lib.bce_fwd_grad_loss_acc(
            workspace.d_fc_out,
            workspace.d_y,
            workspace.d_grad_logits,
            workspace.d_loss_sum,
            workspace.d_correct,
            batch_size,
        )
    else:
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


def backward_fc_impl(
    runtime: "CudaRuntimeState",
    workspace: "BatchWorkspace",
    arch: "CudaNetGeometry",
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


def backward_convs_impl(
    runtime: "CudaRuntimeState",
    workspace: "BatchWorkspace",
    arch: "CudaNetGeometry",
    batch_size: int,
) -> None:
    lib.clip_inplace(workspace.d_pre_fc_grad_nchw, _C_GRAD_POOL_CLIP, batch_size * arch.fc_in)
    grad_nchw = workspace.d_pre_fc_grad_nchw

    for i in reversed(range(arch.n_conv)):
        stage = arch.conv_stages[i]
        if stage.layer_norm and not stage.pool:
            ln_idx = arch.ln_param_idx(i)
            lib.layer_norm_backward(
                workspace.d_ln_input_grad[i],
                grad_nchw,
                workspace.d_conv_nchw[i],
                runtime.weights.ln_gamma[ln_idx],
                batch_size, stage.out_c, stage.h_out, stage.w_out,
                c_float(1e-5),
            )
            grad_for_cnhw = workspace.d_ln_input_grad[i]
        elif stage.batch_norm and not stage.pool:
            bn_idx = arch.bn_param_idx(i)
            zero_floats(workspace.d_bn_dgamma[i], stage.out_c)
            zero_floats(workspace.d_bn_dbeta[i], stage.out_c)
            lib.bn_backward(
                workspace.d_bn_input_grad[i],
                workspace.d_bn_dgamma[i],
                workspace.d_bn_dbeta[i],
                grad_nchw,
                workspace.d_bn_x_hat[i],
                runtime.weights.bn_gamma[bn_idx],
                workspace.d_bn_inv_std[i],
                batch_size, stage.out_c, stage.h_out, stage.w_out,
            )
            grad_for_cnhw = workspace.d_bn_input_grad[i]
        else:
            grad_for_cnhw = grad_nchw

        if stage.pool:
            nchw_to_cnhw_into(
                grad_for_cnhw,
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
                grad_for_cnhw,
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
