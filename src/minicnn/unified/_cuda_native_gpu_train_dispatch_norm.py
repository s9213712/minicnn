from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.gpu_training import (
    native_gpu_batchnorm_linear_training_step,
    native_gpu_groupnorm_linear_training_step,
    native_gpu_layernorm_linear_training_step,
    native_gpu_layernorm2d_linear_training_step,
)
from minicnn.unified._cuda_native_context import NativeTrainingContext


def run_gpu_native_norm_batch(
    ctx: NativeTrainingContext,
    *,
    optimizer_view: Any,
    optimizer_state: dict[str, Any],
    gpu_training_plan: dict[str, Any],
    params: dict[str, np.ndarray],
    xb: np.ndarray,
    yb: np.ndarray,
) -> tuple[dict[str, np.ndarray], Any] | None:
    velocity_state = optimizer_state.setdefault('velocity', {})
    kind = gpu_training_plan['kind']
    if kind == 'batchnorm_linear':
        bn_node = gpu_training_plan['batchnorm_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        bn_weight_key = f'_w_{bn_node.name}'
        bn_bias_key = f'_b_{bn_node.name}'
        running_mean_key = f'_running_mean_{bn_node.name}'
        running_var_key = f'_running_var_{bn_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_batchnorm_linear_training_step(
            xb,
            yb,
            params[bn_weight_key],
            params[bn_bias_key],
            params[running_mean_key],
            params[running_var_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            bn_eps=float(bn_node.attrs.get('eps', 1e-5)),
            bn_momentum=float(bn_node.attrs.get('momentum', 0.1)),
            bn_weight_velocity=velocity_state.get(bn_weight_key),
            bn_bias_velocity=velocity_state.get(bn_bias_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[bn_weight_key] = step.updated_bn_weight
        params[bn_bias_key] = step.updated_bn_bias
        params[running_mean_key] = step.updated_running_mean
        params[running_var_key] = step.updated_running_var
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[bn_weight_key] = step.updated_bn_weight_velocity
        velocity_state[bn_bias_key] = step.updated_bn_bias_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind == 'layernorm2d_linear':
        norm_node = gpu_training_plan['layernorm2d_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        norm_weight_key = f'_w_{norm_node.name}'
        norm_bias_key = f'_b_{norm_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_layernorm2d_linear_training_step(
            xb,
            yb,
            params[norm_weight_key],
            params[norm_bias_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
            norm_weight_velocity=velocity_state.get(norm_weight_key),
            norm_bias_velocity=velocity_state.get(norm_bias_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[norm_weight_key] = step.updated_norm_weight
        params[norm_bias_key] = step.updated_norm_bias
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind in {'layernorm_linear', 'layernorm_activation_linear'}:
        norm_node = gpu_training_plan['layernorm_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        norm_weight_key = f'_w_{norm_node.name}'
        norm_bias_key = f'_b_{norm_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_layernorm_linear_training_step(
            xb,
            yb,
            params[norm_weight_key],
            params[norm_bias_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            activation=(
                str(gpu_training_plan['activation_node'].op_type)
                if kind == 'layernorm_activation_linear' else None
            ),
            activation_alpha=(
                float(gpu_training_plan['activation_node'].attrs.get('negative_slope', 0.01))
                if kind == 'layernorm_activation_linear' else 0.01
            ),
            normalized_shape=norm_node.attrs.get('normalized_shape'),
            norm_eps=float(norm_node.attrs.get('eps', 1e-5)),
            norm_weight_velocity=velocity_state.get(norm_weight_key),
            norm_bias_velocity=velocity_state.get(norm_bias_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            bound_lib=ctx.device_runtime.bound_lib,
        )
        params = dict(params)
        params[norm_weight_key] = step.updated_norm_weight
        params[norm_bias_key] = step.updated_norm_bias
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind == 'groupnorm_linear':
        norm_node = gpu_training_plan['groupnorm_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        norm_weight_key = f'_w_{norm_node.name}'
        norm_bias_key = f'_b_{norm_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_groupnorm_linear_training_step(
            xb,
            yb,
            params[norm_weight_key],
            params[norm_bias_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            num_groups=int(norm_node.attrs.get('num_groups', 1)),
            norm_eps=float(norm_node.attrs.get('eps', 1e-5)),
            norm_weight_velocity=velocity_state.get(norm_weight_key),
            norm_bias_velocity=velocity_state.get(norm_bias_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[norm_weight_key] = step.updated_norm_weight
        params[norm_bias_key] = step.updated_norm_bias
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    return None
