"""Minimal training utilities for cuda_native.

Provides a single-step training function that wires together:
  forward with cache → loss → backward → SGD parameter update

This module is intentionally small.  Schedulers, weight decay, and
multi-epoch loops are out of scope for Phase 3.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from minicnn.cuda_native.backward import BackwardExecutor, make_default_backward_registry
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.loss import bce_with_logits_loss, cross_entropy_loss, mse_loss


LossType = Literal['cross_entropy', 'mse', 'bce_with_logits']
OptimizerType = Literal['sgd', 'adam', 'adamw', 'rmsprop']


def _clip_gradients(
    param_grads: dict[str, np.ndarray],
    max_norm: float,
    *,
    param_keys: set[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    if param_keys is None:
        selected_keys = tuple(param_grads.keys())
    else:
        selected_keys = tuple(key for key in param_keys if key in param_grads)
    if not selected_keys:
        return {}
    if max_norm <= 0.0:
        return {key: param_grads[key] for key in selected_keys}
    total_sq_norm = 0.0
    for key in selected_keys:
        grad = param_grads[key]
        total_sq_norm += float(np.sum(np.square(grad, dtype=np.float32)))
    total_norm = float(np.sqrt(total_sq_norm))
    if total_norm <= max_norm or total_norm == 0.0:
        return {key: param_grads[key] for key in selected_keys}
    scale = max_norm / (total_norm + 1e-12)
    return {
        key: (param_grads[key] * scale).astype(np.float32)
        for key in selected_keys
    }


def sgd_update(
    params: dict[str, np.ndarray],
    param_grads: dict[str, np.ndarray],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    optimizer_state: dict[str, Any] | None = None,
    grad_clip_global: float = 0.0,
    param_keys: set[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return a new param dict after one SGD step.

    Does not mutate the input dicts.
    """
    clipped_grads = _clip_gradients(param_grads, grad_clip_global, param_keys=param_keys)
    velocity: dict[str, np.ndarray] = {}
    runtime = _optimizer_runtime_state(optimizer_state, optimizer_type='sgd')
    if optimizer_state is not None:
        velocity = optimizer_state.setdefault('velocity', {})

    updated: dict[str, np.ndarray] = {}
    for key, val in params.items():
        if key in clipped_grads:
            g = clipped_grads[key]
            if weight_decay > 0.0:
                g = g + weight_decay * val
            if momentum > 0.0:
                prev_v = _get_or_init_state_tensor(velocity, key, val.astype(np.float32), runtime)
                prev_v *= momentum
                prev_v -= lr * g
                updated[key] = (val + prev_v).astype(np.float32)
            else:
                updated[key] = (val - lr * g).astype(np.float32)
        else:
            updated[key] = val
    return updated


def adamw_update(
    params: dict[str, np.ndarray],
    param_grads: dict[str, np.ndarray],
    lr: float,
    weight_decay: float = 0.01,
    optimizer_state: dict[str, Any] | None = None,
    grad_clip_global: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    param_keys: set[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    clipped_grads = _clip_gradients(param_grads, grad_clip_global, param_keys=param_keys)
    state = optimizer_state if optimizer_state is not None else {}
    runtime = _optimizer_runtime_state(optimizer_state, optimizer_type='adamw')
    m_state: dict[str, np.ndarray] = state.setdefault('adamw_m', {})
    v_state: dict[str, np.ndarray] = state.setdefault('adamw_v', {})
    step = int(state.get('adamw_step', 0)) + 1
    state['adamw_step'] = step

    updated: dict[str, np.ndarray] = {}
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    for key, val in params.items():
        if key not in clipped_grads:
            updated[key] = val
            continue
        grad = np.asarray(clipped_grads[key], dtype=np.float32)
        val_f32 = np.asarray(val, dtype=np.float32)
        m_prev = _get_or_init_state_tensor(m_state, key, val_f32, runtime)
        v_prev = _get_or_init_state_tensor(v_state, key, val_f32, runtime)
        grad_sq = np.square(grad, dtype=np.float32)
        m_prev *= beta1
        m_prev += (1.0 - beta1) * grad
        v_prev *= beta2
        v_prev += (1.0 - beta2) * grad_sq
        m_hat = m_prev / bias_correction1
        v_hat = v_prev / bias_correction2
        next_val = val_f32.copy()
        if weight_decay > 0.0:
            next_val = next_val - lr * weight_decay * next_val
        next_val = next_val - lr * (m_hat / (np.sqrt(v_hat) + eps))
        updated[key] = next_val
    return updated


def adam_update(
    params: dict[str, np.ndarray],
    param_grads: dict[str, np.ndarray],
    lr: float,
    weight_decay: float = 0.0,
    optimizer_state: dict[str, Any] | None = None,
    grad_clip_global: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    param_keys: set[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    clipped_grads = _clip_gradients(param_grads, grad_clip_global, param_keys=param_keys)
    state = optimizer_state if optimizer_state is not None else {}
    m_state: dict[str, np.ndarray] = state.setdefault('adam_m', {})
    v_state: dict[str, np.ndarray] = state.setdefault('adam_v', {})
    runtime = _optimizer_runtime_state(optimizer_state, optimizer_type='adam')
    step = int(state.get('adam_step', 0)) + 1
    state['adam_step'] = step

    updated: dict[str, np.ndarray] = {}
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    for key, val in params.items():
        if key not in clipped_grads:
            updated[key] = val
            continue
        grad = clipped_grads[key].astype(np.float32)
        if weight_decay > 0.0:
            grad = grad + weight_decay * val.astype(np.float32)
        val_f32 = val.astype(np.float32)
        m_prev = _get_or_init_state_tensor(m_state, key, val_f32, runtime)
        v_prev = _get_or_init_state_tensor(v_state, key, val_f32, runtime)
        m_prev *= beta1
        m_prev += (1.0 - beta1) * grad
        v_prev *= beta2
        v_prev += (1.0 - beta2) * np.square(grad, dtype=np.float32)
        m_hat = m_prev / bias_correction1
        v_hat = v_prev / bias_correction2
        updated[key] = (val.astype(np.float32) - lr * (m_hat / (np.sqrt(v_hat) + eps))).astype(np.float32)
    return updated


def rmsprop_update(
    params: dict[str, np.ndarray],
    param_grads: dict[str, np.ndarray],
    lr: float,
    weight_decay: float = 0.0,
    optimizer_state: dict[str, Any] | None = None,
    grad_clip_global: float = 0.0,
    alpha: float = 0.99,
    eps: float = 1e-8,
    momentum: float = 0.0,
    param_keys: set[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    clipped_grads = _clip_gradients(param_grads, grad_clip_global, param_keys=param_keys)
    state = optimizer_state if optimizer_state is not None else {}
    v_state: dict[str, np.ndarray] = state.setdefault('rmsprop_v', {})
    buf_state: dict[str, np.ndarray] = state.setdefault('rmsprop_buf', {})
    runtime = _optimizer_runtime_state(optimizer_state, optimizer_type='rmsprop')
    updated: dict[str, np.ndarray] = {}
    for key, val in params.items():
        if key not in clipped_grads:
            updated[key] = val
            continue
        grad = clipped_grads[key].astype(np.float32)
        if weight_decay > 0.0:
            grad = grad + weight_decay * val.astype(np.float32)
        val_f32 = val.astype(np.float32)
        v_prev = _get_or_init_state_tensor(v_state, key, val_f32, runtime)
        v_prev *= alpha
        v_prev += (1.0 - alpha) * np.square(grad, dtype=np.float32)
        step_grad = (grad / (np.sqrt(v_prev) + eps)).astype(np.float32)
        if momentum > 0.0:
            buf_prev = _get_or_init_state_tensor(buf_state, key, val_f32, runtime)
            buf_prev *= momentum
            buf_prev += step_grad
            update = buf_prev
        else:
            update = step_grad
        updated[key] = (val.astype(np.float32) - lr * update).astype(np.float32)
    return updated


def _accumulate_param_grads(
    grad_buffer: dict[str, np.ndarray],
    param_grads: dict[str, np.ndarray],
    runtime: dict[str, Any] | None = None,
    active_keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
    for key, grad in param_grads.items():
        grad_f32 = np.asarray(grad, dtype=np.float32)
        if active_keys is not None:
            active_keys.add(key)
        if key in grad_buffer:
            np.add(grad_buffer[key], grad_f32, out=grad_buffer[key], casting='unsafe')
            if runtime is not None:
                runtime['grad_buffer_reuses'] = int(runtime.get('grad_buffer_reuses', 0)) + 1
        else:
            grad_buffer[key] = grad_f32.copy()
            if runtime is not None:
                runtime['grad_buffer_allocations'] = int(runtime.get('grad_buffer_allocations', 0)) + 1
    return grad_buffer


def _reset_grad_buffer(
    grad_buffer: dict[str, np.ndarray],
    runtime: dict[str, Any] | None = None,
    active_keys: set[str] | None = None,
) -> None:
    if active_keys is not None:
        keys = list(active_keys)
    else:
        keys = list(grad_buffer.keys())
    if not keys:
        return
    zeroed_tensors = 0
    for key in keys:
        value = grad_buffer.get(key)
        if isinstance(value, np.ndarray):
            value.fill(0.0)
            zeroed_tensors += 1
    if active_keys is not None:
        active_keys.clear()
    else:
        grad_buffer.clear()
    if runtime is not None:
        runtime['grad_buffer_reset_events'] = int(runtime.get('grad_buffer_reset_events', 0)) + 1
        runtime['grad_buffer_zeroed_tensors'] = int(runtime.get('grad_buffer_zeroed_tensors', 0)) + zeroed_tensors


def _all_finite_tensors(tensors: dict[str, np.ndarray]) -> bool:
    for value in tensors.values():
        if not np.all(np.isfinite(value)):
            return False
    return True


def _amp_cache_signature(params: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    signature: list[tuple[str, tuple[int, ...], str]] = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, np.ndarray):
            signature.append((key, tuple(value.shape), str(value.dtype)))
        else:
            signature.append((key, (), type(value).__name__))
    return tuple(signature)


def _prepare_amp_params(
    params: dict[str, np.ndarray],
    amp_enabled: bool,
    optimizer_state: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if not amp_enabled:
        return params
    state = optimizer_state if optimizer_state is not None else {}
    amp_state = state.setdefault('amp', {})
    cache_key = _amp_cache_signature(params)
    cached_key = amp_state.get('params_cache_key')
    cached_params = amp_state.get('params_fp16_cache')
    if cached_key == cache_key and isinstance(cached_params, dict):
        amp_state['cache_hits'] = int(amp_state.get('cache_hits', 0)) + 1
        for key, value in params.items():
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.floating)
                and key in cached_params
                and isinstance(cached_params[key], np.ndarray)
                and cached_params[key].shape == value.shape
                and cached_params[key].dtype == np.float16
            ):
                np.copyto(cached_params[key], value, casting='unsafe')
        amp_state['cache_updates'] = int(amp_state.get('cache_updates', 0)) + 1
        return cached_params
    fp16_params: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            fp16_params[key] = value.astype(np.float16)
        else:
            fp16_params[key] = value
    amp_state['params_cache_key'] = cache_key
    amp_state['params_fp16_cache'] = fp16_params
    amp_state['cache_allocations'] = int(amp_state.get('cache_allocations', 0)) + 1
    return fp16_params


def _optimizer_runtime_state(
    optimizer_state: dict[str, Any] | None,
    *,
    optimizer_type: str,
) -> dict[str, Any]:
    state = optimizer_state if optimizer_state is not None else {}
    runtime = state.setdefault('optimizer_runtime', {})
    runtime['optimizer_type'] = optimizer_type
    runtime['steps'] = int(runtime.get('steps', 0)) + 1
    runtime.setdefault('state_tensor_allocations', 0)
    runtime.setdefault('state_tensor_updates', 0)
    runtime.setdefault('grad_buffer_allocations', 0)
    runtime.setdefault('grad_buffer_reuses', 0)
    runtime.setdefault('grad_buffer_reset_events', 0)
    runtime.setdefault('grad_buffer_zeroed_tensors', 0)
    return runtime


def _get_or_init_state_tensor(
    bucket: dict[str, np.ndarray],
    key: str,
    like: np.ndarray,
    runtime: dict[str, Any],
) -> np.ndarray:
    current = bucket.get(key)
    if isinstance(current, np.ndarray) and current.shape == like.shape and current.dtype == np.float32:
        runtime['state_tensor_updates'] = int(runtime.get('state_tensor_updates', 0)) + 1
        return current
    current = np.zeros_like(like, dtype=np.float32)
    bucket[key] = current
    runtime['state_tensor_allocations'] = int(runtime.get('state_tensor_allocations', 0)) + 1
    return current


def train_step(
    graph: NativeGraph,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    lr: float,
    loss_type: LossType = 'cross_entropy',
    optimizer_type: OptimizerType = 'sgd',
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    label_smoothing: float = 0.0,
    grad_accum_steps: int = 1,
    apply_optimizer_step: bool = True,
    amp_enabled: bool = False,
    amp_loss_scale: float = 128.0,
    amp_dynamic_scale: bool = True,
    amp_scale_growth: float = 2.0,
    amp_scale_backoff: float = 0.5,
    amp_scale_window: int = 200,
    optimizer_state: dict[str, Any] | None = None,
    grad_clip_global: float = 0.0,
    fwd_executor: ForwardExecutor | None = None,
    bwd_executor: BackwardExecutor | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    """Execute one forward-backward-update cycle.

    Args:
        graph:      NativeGraph built with build_graph().
        x:          Input batch, shape matching graph.input_spec.shape.
        y:          Labels — integer class indices (N,) for cross_entropy,
                    or float targets for mse.
        params:     Weight dict {'_w_{node}': ..., '_b_{node}': ...}.
        lr:         Learning rate for SGD.
        loss_type:  'cross_entropy' or 'mse'.
        weight_decay: L2 regularisation coefficient.
        fwd_executor: optional pre-built ForwardExecutor.
        bwd_executor: optional pre-built BackwardExecutor.

    Returns:
        (loss_value, updated_params)
    """
    if graph.input_spec is None:
        raise ValueError('Graph has no input_spec')

    fwd = fwd_executor or ForwardExecutor()
    bwd = bwd_executor or BackwardExecutor()
    state = optimizer_state if optimizer_state is not None else {}
    runtime = _optimizer_runtime_state(optimizer_state, optimizer_type=optimizer_type)
    amp_state = state.setdefault('amp', {})
    active_loss_scale = float(
        amp_state.get('loss_scale', amp_loss_scale if amp_loss_scale > 0.0 else 128.0)
    )
    amp_state.setdefault('good_steps', 0)
    amp_state.setdefault('skipped_steps', 0)
    amp_state.setdefault('finite_steps', 0)
    amp_state.setdefault('overflow_steps', 0)
    amp_state.setdefault('cache_hits', 0)
    amp_state.setdefault('cache_updates', 0)
    amp_state.setdefault('cache_allocations', 0)
    grad_buffer = state.setdefault('grad_buffer', {})
    grad_buffer_active_keys = state.setdefault('grad_buffer_active_keys', set())

    # Forward pass — cache activations for backward
    fwd_input = x.astype(np.float16) if amp_enabled else x
    fwd_params = _prepare_amp_params(params, amp_enabled, optimizer_state)
    ctx, cache = fwd.run_with_cache(
        graph,
        {graph.input_spec.name: fwd_input},
        params=fwd_params,
        mode='train',
    )

    # Loss + initial gradient
    out_name = graph.output_spec.name if graph.output_spec else graph.nodes[-1].outputs[0]
    logits = ctx[out_name]

    if loss_type == 'cross_entropy':
        loss_val, grad_logits = cross_entropy_loss(
            logits,
            y,
            label_smoothing=label_smoothing,
        )
    elif loss_type == 'bce_with_logits':
        loss_val, grad_logits = bce_with_logits_loss(logits, y)
    elif loss_type == 'mse':
        loss_val, grad_logits = mse_loss(logits, y.astype(np.float32))
    else:
        raise ValueError(
            f'Unknown loss_type: {loss_type!r}. Choose "cross_entropy", "bce_with_logits", or "mse".'
        )
    if amp_enabled:
        grad_logits = (grad_logits * active_loss_scale).astype(np.float32)
    if grad_accum_steps > 1:
        grad_logits = (grad_logits / float(grad_accum_steps)).astype(np.float32)

    # Backward pass
    _grad_input, param_grads = bwd.run(graph, grad_logits, cache)
    if amp_enabled:
        scaled_param_grads = {
            key: (grad / active_loss_scale).astype(np.float32)
            for key, grad in param_grads.items()
        }
    else:
        scaled_param_grads = param_grads

    if amp_enabled and not _all_finite_tensors(scaled_param_grads):
        _reset_grad_buffer(grad_buffer, runtime, active_keys=grad_buffer_active_keys)
        amp_state['skipped_steps'] = int(amp_state.get('skipped_steps', 0)) + 1
        amp_state['overflow_steps'] = int(amp_state.get('overflow_steps', 0)) + 1
        amp_state['good_steps'] = 0
        if amp_dynamic_scale:
            amp_state['loss_scale'] = max(1.0, active_loss_scale * amp_scale_backoff)
        else:
            amp_state['loss_scale'] = active_loss_scale
        return loss_val, dict(params)

    _accumulate_param_grads(
        grad_buffer,
        scaled_param_grads,
        runtime,
        active_keys=grad_buffer_active_keys,
    )
    if amp_enabled:
        amp_state['finite_steps'] = int(amp_state.get('finite_steps', 0)) + 1

    active_grad_keys = tuple(grad_buffer_active_keys)

    if not apply_optimizer_step:
        updated_params = dict(params)
    elif optimizer_type == 'sgd':
        updated_params = sgd_update(
            params,
            grad_buffer,
            lr,
            weight_decay,
            momentum=momentum,
            optimizer_state=optimizer_state,
            grad_clip_global=grad_clip_global,
            param_keys=active_grad_keys,
        )
        _reset_grad_buffer(grad_buffer, runtime, active_keys=grad_buffer_active_keys)
    elif optimizer_type == 'adamw':
        updated_params = adamw_update(
            params,
            grad_buffer,
            lr,
            weight_decay=weight_decay,
            optimizer_state=optimizer_state,
            grad_clip_global=grad_clip_global,
            param_keys=active_grad_keys,
        )
        _reset_grad_buffer(grad_buffer, runtime, active_keys=grad_buffer_active_keys)
    elif optimizer_type == 'adam':
        updated_params = adam_update(
            params,
            grad_buffer,
            lr,
            weight_decay=weight_decay,
            optimizer_state=optimizer_state,
            grad_clip_global=grad_clip_global,
            param_keys=active_grad_keys,
        )
        _reset_grad_buffer(grad_buffer, runtime, active_keys=grad_buffer_active_keys)
    elif optimizer_type == 'rmsprop':
        updated_params = rmsprop_update(
            params,
            grad_buffer,
            lr,
            weight_decay=weight_decay,
            optimizer_state=optimizer_state,
            grad_clip_global=grad_clip_global,
            momentum=momentum,
            param_keys=active_grad_keys,
        )
        _reset_grad_buffer(grad_buffer, runtime, active_keys=grad_buffer_active_keys)
    else:
        raise ValueError(
            f'Unknown optimizer_type: {optimizer_type!r}. Choose "sgd", "adam", "adamw", or "rmsprop".'
        )

    if amp_enabled and apply_optimizer_step:
        amp_state['good_steps'] = int(amp_state.get('good_steps', 0)) + 1
        if amp_dynamic_scale and int(amp_state['good_steps']) >= max(1, int(amp_scale_window)):
            amp_state['loss_scale'] = active_loss_scale * amp_scale_growth
            amp_state['good_steps'] = 0
        else:
            amp_state['loss_scale'] = active_loss_scale

    return loss_val, updated_params
