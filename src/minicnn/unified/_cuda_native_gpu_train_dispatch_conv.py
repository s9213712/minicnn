from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.gpu_training import (
    native_gpu_conv_linear_training_step,
    native_gpu_depthwise_layernorm2d_linear_training_step,
    native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step,
    native_gpu_depthwise_layernorm2d_pointwise_linear_training_step,
    native_gpu_two_conv_relu_pool_linear_training_step,
)
from minicnn.unified._cuda_native_context import NativeTrainingContext


def run_gpu_native_conv_batch(
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
    if kind == 'depthwise_layernorm2d_linear':
        conv_node = gpu_training_plan['conv_node']
        norm_node = gpu_training_plan['layernorm2d_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        conv_weight_key = f'_w_{conv_node.name}'
        norm_weight_key = f'_w_{norm_node.name}'
        norm_bias_key = f'_b_{norm_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_depthwise_layernorm2d_linear_training_step(
            xb,
            yb,
            params[conv_weight_key],
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
            conv_weight_velocity=velocity_state.get(conv_weight_key),
            norm_weight_velocity=velocity_state.get(norm_weight_key),
            norm_bias_velocity=velocity_state.get(norm_bias_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[conv_weight_key] = step.updated_conv_weight
        params[norm_weight_key] = step.updated_norm_weight
        params[norm_bias_key] = step.updated_norm_bias
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[conv_weight_key] = step.updated_conv_weight_velocity
        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind == 'depthwise_layernorm2d_pointwise_linear':
        depthwise_node = gpu_training_plan['depthwise_node']
        norm_node = gpu_training_plan['layernorm2d_node']
        pointwise_node = gpu_training_plan['pointwise_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        depthwise_weight_key = f'_w_{depthwise_node.name}'
        norm_weight_key = f'_w_{norm_node.name}'
        norm_bias_key = f'_b_{norm_node.name}'
        pointwise_weight_key = f'_w_{pointwise_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_depthwise_layernorm2d_pointwise_linear_training_step(
            xb,
            yb,
            params[depthwise_weight_key],
            params[norm_weight_key],
            params[norm_bias_key],
            params[pointwise_weight_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
            depthwise_weight_velocity=velocity_state.get(depthwise_weight_key),
            norm_weight_velocity=velocity_state.get(norm_weight_key),
            norm_bias_velocity=velocity_state.get(norm_bias_key),
            pointwise_weight_velocity=velocity_state.get(pointwise_weight_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[depthwise_weight_key] = step.updated_depthwise_weight
        params[norm_weight_key] = step.updated_norm_weight
        params[norm_bias_key] = step.updated_norm_bias
        params[pointwise_weight_key] = step.updated_pointwise_weight
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[depthwise_weight_key] = step.updated_depthwise_weight_velocity
        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
        velocity_state[pointwise_weight_key] = step.updated_pointwise_weight_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind == 'depthwise_layernorm2d_pointwise_gelu_pointwise_linear':
        depthwise_node = gpu_training_plan['depthwise_node']
        norm_node = gpu_training_plan['layernorm2d_node']
        pointwise1_node = gpu_training_plan['pointwise1_node']
        pointwise2_node = gpu_training_plan['pointwise2_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        depthwise_weight_key = f'_w_{depthwise_node.name}'
        norm_weight_key = f'_w_{norm_node.name}'
        norm_bias_key = f'_b_{norm_node.name}'
        pointwise1_weight_key = f'_w_{pointwise1_node.name}'
        pointwise2_weight_key = f'_w_{pointwise2_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
            xb,
            yb,
            params[depthwise_weight_key],
            params[norm_weight_key],
            params[norm_bias_key],
            params[pointwise1_weight_key],
            params[pointwise2_weight_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
            depthwise_weight_velocity=velocity_state.get(depthwise_weight_key),
            norm_weight_velocity=velocity_state.get(norm_weight_key),
            norm_bias_velocity=velocity_state.get(norm_bias_key),
            pointwise1_weight_velocity=velocity_state.get(pointwise1_weight_key),
            pointwise2_weight_velocity=velocity_state.get(pointwise2_weight_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            activation_kind=str(gpu_training_plan.get('activation_kind', 'GELU')),
            activation_alpha=float(gpu_training_plan.get('activation_alpha', 0.01)),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[depthwise_weight_key] = step.updated_depthwise_weight
        params[norm_weight_key] = step.updated_norm_weight
        params[norm_bias_key] = step.updated_norm_bias
        params[pointwise1_weight_key] = step.updated_pointwise1_weight
        params[pointwise2_weight_key] = step.updated_pointwise2_weight
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[depthwise_weight_key] = step.updated_depthwise_weight_velocity
        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
        velocity_state[pointwise1_weight_key] = step.updated_pointwise1_weight_velocity
        velocity_state[pointwise2_weight_key] = step.updated_pointwise2_weight_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind == 'conv_linear':
        conv_node = gpu_training_plan['conv_node']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        conv_weight_key = f'_w_{conv_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_conv_linear_training_step(
            xb,
            yb,
            params[conv_weight_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            conv_weight_velocity=velocity_state.get(conv_weight_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            apply_relu_activation=bool(gpu_training_plan.get('apply_relu_activation', False)),
            activation_kind=gpu_training_plan.get('activation_kind'),
            activation_alpha=float(gpu_training_plan.get('activation_alpha', 0.01)),
            apply_maxpool=bool(gpu_training_plan.get('apply_maxpool', False)),
            conv_kind=str(gpu_training_plan.get('conv_kind', 'conv2d')),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[conv_weight_key] = step.updated_conv_weight
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[conv_weight_key] = step.updated_conv_weight_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    if kind == 'two_conv_relu_pool_linear':
        conv1_node, conv2_node = gpu_training_plan['conv_nodes']
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        conv1_weight_key = f'_w_{conv1_node.name}'
        conv2_weight_key = f'_w_{conv2_node.name}'
        linear_weight_key = f'_w_{gpu_linear_node.name}'
        linear_bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_two_conv_relu_pool_linear_training_step(
            xb,
            yb,
            params[conv1_weight_key],
            params[conv2_weight_key],
            params[linear_weight_key],
            params[linear_bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            conv1_weight_velocity=velocity_state.get(conv1_weight_key),
            conv2_weight_velocity=velocity_state.get(conv2_weight_key),
            linear_weight_velocity=velocity_state.get(linear_weight_key),
            linear_bias_velocity=velocity_state.get(linear_bias_key),
            activation_kind=gpu_training_plan.get('activation_kind'),
            activation_alpha=float(gpu_training_plan.get('activation_alpha', 0.01)),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[conv1_weight_key] = step.updated_conv1_weight
        params[conv2_weight_key] = step.updated_conv2_weight
        params[linear_weight_key] = step.updated_linear_weight
        params[linear_bias_key] = step.updated_linear_bias
        velocity_state[conv1_weight_key] = step.updated_conv1_weight_velocity
        velocity_state[conv2_weight_key] = step.updated_conv2_weight_velocity
        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
        return params, step
    return None
