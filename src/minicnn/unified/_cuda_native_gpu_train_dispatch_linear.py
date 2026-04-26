from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_training import (
    native_gpu_avgpool_linear_training_step,
    native_gpu_global_avgpool_linear_training_step,
    native_gpu_linear_training_step,
    native_gpu_pool_linear_training_step,
    native_gpu_two_linear_relu_training_step,
)
from minicnn.unified._cuda_native_context import NativeTrainingContext


_RUNTIME_DELTA_KEYS = {
    'host_to_device_transfer_events',
    'host_to_device_transfer_bytes',
    'device_to_host_transfer_events',
    'device_to_host_transfer_bytes',
    'allocation_events',
    'allocated_bytes',
    'synchronization_events',
    'device_pointer_allocation_events',
    'device_pointer_free_events',
    'device_pointer_bytes',
    'device_pointer_live_bytes',
    'device_sync_to_host_events',
    'device_sync_to_device_events',
    'persistent_device_cache_hits',
    'persistent_device_cache_misses',
    'persistent_device_cache_invalidations',
    'execution_events',
    'executed_node_count',
}


def _runtime_summary_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta = dict(after)
    for key in _RUNTIME_DELTA_KEYS:
        delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    before_kinds = dict(before.get('execution_kinds', {}))
    delta['execution_kinds'] = {
        str(kind): int(count) - int(before_kinds.get(kind, 0))
        for kind, count in dict(after.get('execution_kinds', {})).items()
        if int(count) - int(before_kinds.get(kind, 0)) != 0
    }
    before_trace_len = len(list(before.get('execution_trace', [])))
    delta['execution_trace'] = list(after.get('execution_trace', []))[before_trace_len:]
    delta['persistent_device_cache_entries'] = int(after.get('persistent_device_cache_entries', 0))
    delta['persistent_device_cache_bytes'] = int(after.get('persistent_device_cache_bytes', 0))
    delta['synchronization_reasons'] = list(after.get('synchronization_reasons', []))[len(list(before.get('synchronization_reasons', []))):]
    return delta


def _persistent_linear_runtime(
    ctx: NativeTrainingContext,
    optimizer_state: dict[str, Any],
) -> DeviceRuntime:
    runtime = optimizer_state.get('persistent_device_runtime')
    if not isinstance(runtime, DeviceRuntime):
        runtime = DeviceRuntime(
            execution_mode='gpu_native',
            tensor_execution_device='gpu',
            bound_lib=ctx.device_runtime.bound_lib,
        )
        optimizer_state['persistent_device_runtime'] = runtime
    runtime.bound_lib = ctx.device_runtime.bound_lib
    return runtime


def run_gpu_native_linear_or_pool_batch(
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
    flat_xb = xb.reshape(xb.shape[0], -1)
    kind = gpu_training_plan['kind']
    if kind == 'linear':
        persistent_runtime = _persistent_linear_runtime(ctx, optimizer_state)
        runtime_before = persistent_runtime.summary()
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        weight_key = f'_w_{gpu_linear_node.name}'
        bias_key = f'_b_{gpu_linear_node.name}'
        step = native_gpu_linear_training_step(
            flat_xb,
            yb,
            params[weight_key],
            params[bias_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            loss_type=ctx.loss_type,
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            optimizer_type=ctx.optimizer_type,
            weight_decay=float(ctx.weight_decay),
            grad_clip_value=float(ctx.grad_clip_global),
            beta1=float(ctx.optimizer_cfg.get('beta1', 0.9)),
            beta2=float(ctx.optimizer_cfg.get('beta2', 0.999)),
            eps=float(ctx.optimizer_cfg.get('eps', 1e-8)),
            rmsprop_alpha=float(ctx.optimizer_cfg.get('alpha', 0.99)),
            step_index=int(optimizer_state.get('optimizer_runtime', {}).get('steps', 0)) + 1,
            weight_velocity=velocity_state.get(weight_key),
            bias_velocity=velocity_state.get(bias_key),
            weight_m=(
                optimizer_state.setdefault('adamw_m' if ctx.optimizer_type == 'adamw' else 'adam_m', {}).get(weight_key)
                if ctx.optimizer_type in {'adam', 'adamw'} else None
            ),
            weight_v=(
                optimizer_state.setdefault('adamw_v' if ctx.optimizer_type == 'adamw' else 'adam_v', {}).get(weight_key)
                if ctx.optimizer_type in {'adam', 'adamw'} else None
            ),
            bias_m=(
                optimizer_state.setdefault('adamw_m' if ctx.optimizer_type == 'adamw' else 'adam_m', {}).get(bias_key)
                if ctx.optimizer_type in {'adam', 'adamw'} else None
            ),
            bias_v=(
                optimizer_state.setdefault('adamw_v' if ctx.optimizer_type == 'adamw' else 'adam_v', {}).get(bias_key)
                if ctx.optimizer_type in {'adam', 'adamw'} else None
            ),
            weight_rmsprop_v=(
                optimizer_state.setdefault('rmsprop_v', {}).get(weight_key)
                if ctx.optimizer_type == 'rmsprop' else None
            ),
            weight_rmsprop_buf=(
                optimizer_state.setdefault('rmsprop_buf', {}).get(weight_key)
                if ctx.optimizer_type == 'rmsprop' else None
            ),
            bias_rmsprop_v=(
                optimizer_state.setdefault('rmsprop_v', {}).get(bias_key)
                if ctx.optimizer_type == 'rmsprop' else None
            ),
            bias_rmsprop_buf=(
                optimizer_state.setdefault('rmsprop_buf', {}).get(bias_key)
                if ctx.optimizer_type == 'rmsprop' else None
            ),
            device_runtime=persistent_runtime,
            persistent_device_state=True,
            persistent_cache_prefix=f'train:{weight_key}:{bias_key}:{ctx.optimizer_type}',
            return_intermediates=False,
        )
        step = replace(step, runtime_summary=_runtime_summary_delta(runtime_before, step.runtime_summary))
        params = dict(params)
        params[weight_key] = step.updated_weight
        params[bias_key] = step.updated_bias
        velocity_state[weight_key] = step.updated_weight_velocity
        velocity_state[bias_key] = step.updated_bias_velocity
        if ctx.optimizer_type in {'adam', 'adamw'}:
            m_bucket = optimizer_state.setdefault('adamw_m' if ctx.optimizer_type == 'adamw' else 'adam_m', {})
            v_bucket = optimizer_state.setdefault('adamw_v' if ctx.optimizer_type == 'adamw' else 'adam_v', {})
            m_bucket[weight_key] = step.updated_weight_m
            v_bucket[weight_key] = step.updated_weight_v
            m_bucket[bias_key] = step.updated_bias_m
            v_bucket[bias_key] = step.updated_bias_v
        if ctx.optimizer_type == 'rmsprop':
            v_bucket = optimizer_state.setdefault('rmsprop_v', {})
            buf_bucket = optimizer_state.setdefault('rmsprop_buf', {})
            v_bucket[weight_key] = step.updated_weight_rmsprop_v
            buf_bucket[weight_key] = step.updated_weight_rmsprop_buf
            v_bucket[bias_key] = step.updated_bias_rmsprop_v
            buf_bucket[bias_key] = step.updated_bias_rmsprop_buf
        return params, step
    if kind == 'two_linear_activation':
        first_linear, second_linear = gpu_training_plan['linear_nodes']
        activation_node = gpu_training_plan['activation_node']
        w1_key = f'_w_{first_linear.name}'
        b1_key = f'_b_{first_linear.name}'
        w2_key = f'_w_{second_linear.name}'
        b2_key = f'_b_{second_linear.name}'
        step = native_gpu_two_linear_relu_training_step(
            flat_xb,
            yb,
            params[w1_key],
            params[b1_key],
            params[w2_key],
            params[b2_key],
            lr=float(optimizer_view.lr),
            momentum=float(ctx.momentum),
            grad_clip_value=float(ctx.grad_clip_global),
            weight_decay=float(ctx.weight_decay),
            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
            weight1_velocity=velocity_state.get(w1_key),
            bias1_velocity=velocity_state.get(b1_key),
            weight2_velocity=velocity_state.get(w2_key),
            bias2_velocity=velocity_state.get(b2_key),
            activation=str(activation_node.op_type),
            activation_alpha=float(activation_node.attrs.get('negative_slope', 0.01)),
            bound_lib=ctx.device_runtime.bound_lib,
            return_intermediates=False,
        )
        params = dict(params)
        params[w1_key] = step.updated_weight1
        params[b1_key] = step.updated_bias1
        params[w2_key] = step.updated_weight2
        params[b2_key] = step.updated_bias2
        velocity_state[w1_key] = step.updated_weight1_velocity
        velocity_state[b1_key] = step.updated_bias1_velocity
        velocity_state[w2_key] = step.updated_weight2_velocity
        velocity_state[b2_key] = step.updated_bias2_velocity
        return params, step
    if kind in {'pool_linear', 'global_avgpool_linear', 'avgpool_linear'}:
        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
        weight_key = f'_w_{gpu_linear_node.name}'
        bias_key = f'_b_{gpu_linear_node.name}'
        if kind == 'pool_linear':
            step = native_gpu_pool_linear_training_step(
                xb,
                yb,
                params[weight_key],
                params[bias_key],
                lr=float(optimizer_view.lr),
                momentum=float(ctx.momentum),
                grad_clip_value=float(ctx.grad_clip_global),
                weight_decay=float(ctx.weight_decay),
                label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                weight_velocity=velocity_state.get(weight_key),
                bias_velocity=velocity_state.get(bias_key),
                bound_lib=ctx.device_runtime.bound_lib,
                return_intermediates=False,
            )
        elif kind == 'global_avgpool_linear':
            step = native_gpu_global_avgpool_linear_training_step(
                xb,
                yb,
                params[weight_key],
                params[bias_key],
                lr=float(optimizer_view.lr),
                momentum=float(ctx.momentum),
                grad_clip_value=float(ctx.grad_clip_global),
                weight_decay=float(ctx.weight_decay),
                label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                weight_velocity=velocity_state.get(weight_key),
                bias_velocity=velocity_state.get(bias_key),
                bound_lib=ctx.device_runtime.bound_lib,
                return_intermediates=False,
            )
        else:
            step = native_gpu_avgpool_linear_training_step(
                xb,
                yb,
                params[weight_key],
                params[bias_key],
                lr=float(optimizer_view.lr),
                momentum=float(ctx.momentum),
                grad_clip_value=float(ctx.grad_clip_global),
                weight_decay=float(ctx.weight_decay),
                label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                weight_velocity=velocity_state.get(weight_key),
                bias_velocity=velocity_state.get(bias_key),
                bound_lib=ctx.device_runtime.bound_lib,
                return_intermediates=False,
            )
        params = dict(params)
        params[weight_key] = step.updated_weight
        params[bias_key] = step.updated_bias
        velocity_state[weight_key] = step.updated_weight_velocity
        velocity_state[bias_key] = step.updated_bias_velocity
        return params, step
    return None
