from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_training_common import (
    _apply_global_grad_clip,
    _load_bound_lib,
    _run_softmax_xent_loss,
)
from minicnn.cuda_native.gpu_training_types import (
    NativeGpuLayerNormLinearTrainingStepResult,
    NativeGpuLayerNorm2dLinearTrainingStepResult,
)


def native_gpu_layernorm_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    activation: str | None = None,
    activation_alpha: float = 0.01,
    normalized_shape: int | tuple[int, ...] | list[int] | None = None,
    norm_eps: float = 1e-5,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuLayerNormLinearTrainingStepResult:
    """Run Flatten + LayerNorm(native when available) + optional activation + Linear(native) + SoftmaxCE + SGD."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    flat_x = np.ascontiguousarray(x_f32.reshape(int(x_f32.shape[0]), -1), dtype=np.float32)
    if normalized_shape is None:
        normalized_shape_t = (int(flat_x.shape[1]),)
    elif isinstance(normalized_shape, int):
        normalized_shape_t = (int(normalized_shape),)
    else:
        normalized_shape_t = tuple(int(v) for v in normalized_shape)
    if not normalized_shape_t:
        raise ValueError('native_gpu_layernorm_linear_training_step requires non-empty normalized_shape.')
    if int(np.prod(normalized_shape_t)) != int(flat_x.shape[1]):
        raise ValueError(
            'native_gpu_layernorm_linear_training_step requires normalized_shape product to match flattened features, '
            f'got normalized_shape={normalized_shape_t} and flattened_features={flat_x.shape[1]}.'
        )
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    if norm_weight_f32.shape != normalized_shape_t or norm_bias_f32.shape != normalized_shape_t:
        raise ValueError(
            'native_gpu_layernorm_linear_training_step expects norm weight/bias with shape '
            f'{normalized_shape_t}, got weight={norm_weight_f32.shape}, bias={norm_bias_f32.shape}.'
        )
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_x.shape[1]:
        raise ValueError(
            'native_gpu_layernorm_linear_training_step expects linear_weight with shape '
            f'(out_f, {flat_x.shape[1]}), got {linear_w_f32.shape}.'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_layernorm_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != flat_x.shape[0]:
        raise ValueError('native_gpu_layernorm_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_layernorm_linear_training_step labels must be in [0, out_f).')
    activation_name = None if activation is None else str(activation)
    activation_key = None if activation_name is None else activation_name.lower()
    activation_forward = {
        'relu': ('apply_relu', 'apply_relu'),
        'leakyrelu': ('leaky_relu_forward', 'leaky_relu_forward'),
        'gelu': ('gelu_forward', 'gelu_forward'),
        'silu': ('silu_forward', 'silu_forward'),
        'sigmoid': ('sigmoid_forward', 'sigmoid_forward'),
        'tanh': ('tanh_forward', 'tanh_forward'),
    }
    activation_backward = {
        'relu': ('apply_relu_backward', 'apply_relu_backward'),
        'leakyrelu': ('leaky_relu_backward', 'leaky_relu_backward'),
        'gelu': ('gelu_backward', 'gelu_backward'),
        'silu': ('silu_backward', 'silu_backward'),
        'sigmoid': ('sigmoid_backward', 'sigmoid_backward'),
        'tanh': ('tanh_backward', 'tanh_backward'),
    }
    if activation_key is not None and activation_key not in activation_forward:
        raise ValueError(f'native_gpu_layernorm_linear_training_step does not support activation={activation_name!r}')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))
    use_native_layernorm = hasattr(lib, 'layernorm_nd_forward') and hasattr(lib, 'layernorm_nd_backward')

    out_f = int(linear_w_f32.shape[0])
    tensors = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    input_t = stage(flat_x, 'flatten_output')
    labels_t = stage(labels_i32, 'labels')
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    norm_t = None if not use_native_layernorm else alloc((flat_x.shape[0], flat_x.shape[1]), 'norm_output')
    activated_t = None if activation_key is None else alloc((flat_x.shape[0], flat_x.shape[1]), 'activation_output')
    logits_t = alloc((flat_x.shape[0], out_f), 'logits')
    probs_t = alloc((flat_x.shape[0], out_f), 'probs')
    grad_logits_t = alloc((flat_x.shape[0], out_f), 'grad_logits')
    grad_activation_t = alloc((flat_x.shape[0], flat_x.shape[1]), 'grad_activation_output')
    grad_input_t = None if not use_native_layernorm else alloc((flat_x.shape[0], flat_x.shape[1]), 'grad_input')
    grad_linear_w_t = alloc((out_f, flat_x.shape[1]), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    grad_norm_weight_t = stage(np.zeros_like(norm_weight_f32), 'grad_norm_weight')
    grad_norm_bias_t = stage(np.zeros_like(norm_bias_f32), 'grad_norm_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        if use_native_layernorm:
            lib.layernorm_nd_forward(
                input_t.device_ptr,
                norm_weight_t.device_ptr,
                norm_bias_t.device_ptr,
                norm_t.device_ptr,
                int(flat_x.shape[0]),
                int(flat_x.shape[1]),
                float(norm_eps),
            )
            runtime.record_execution(
                'gpu_native_train:layernorm_forward',
                input_name='flatten_output',
                output_name='norm_output',
                node_count=1,
            )
        else:
            mean = flat_x.mean(axis=1, keepdims=True).astype(np.float32)
            var = flat_x.var(axis=1, keepdims=True).astype(np.float32)
            inv_std = (1.0 / np.sqrt(var + norm_eps)).astype(np.float32)
            norm_output = ((flat_x - mean) * inv_std).astype(np.float32)
            norm_output = norm_output * norm_weight_f32.reshape(1, -1) + norm_bias_f32.reshape(1, -1)
            norm_t = stage(np.ascontiguousarray(norm_output, dtype=np.float32), 'norm_output')
            runtime.record_execution(
                'gpu_native_train:layernorm_forward_reference',
                input_name='flatten_output',
                output_name='norm_output',
                node_count=1,
            )
        dense_input_t = norm_t
        dense_input_name = 'norm_output'
        if activation_key is not None:
            lib.gpu_memcpy_d2d(activated_t.device_ptr, norm_t.device_ptr, norm_t.nbytes)
            forward_symbol, forward_kind = activation_forward[activation_key]
            if activation_key == 'leakyrelu':
                getattr(lib, forward_symbol)(activated_t.device_ptr, float(activation_alpha), int(flat_x.shape[0] * flat_x.shape[1]))
            else:
                getattr(lib, forward_symbol)(activated_t.device_ptr, int(flat_x.shape[0] * flat_x.shape[1]))
            runtime.record_execution(f'gpu_native_train:{forward_kind}', input_name='norm_output', output_name='activation_output', node_count=1)
            dense_input_t = activated_t
            dense_input_name = 'activation_output'
        lib.dense_forward(
            dense_input_t.device_ptr,
            linear_w_t.device_ptr,
            linear_b_t.device_ptr,
            logits_t.device_ptr,
            int(flat_x.shape[0]),
            int(flat_x.shape[1]),
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_forward', input_name=dense_input_name, output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(
            runtime,
            lib,
            logits_t,
            labels_t,
            probs_t,
            grad_logits_t,
            loss_sum_t,
            correct_t,
            int(flat_x.shape[0]),
            out_f,
            label_smoothing=float(label_smoothing),
        )
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(
            grad_logits_t.device_ptr,
            dense_input_t.device_ptr,
            linear_w_t.device_ptr,
            grad_activation_t.device_ptr,
            grad_linear_w_t.device_ptr,
            grad_linear_b_t.device_ptr,
            int(flat_x.shape[0]),
            int(flat_x.shape[1]),
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)

        if activation_key is not None:
            backward_symbol, backward_kind = activation_backward[activation_key]
            if activation_key == 'leakyrelu':
                getattr(lib, backward_symbol)(norm_t.device_ptr, grad_activation_t.device_ptr, float(activation_alpha), int(flat_x.shape[0] * flat_x.shape[1]))
            else:
                getattr(lib, backward_symbol)(norm_t.device_ptr, grad_activation_t.device_ptr, int(flat_x.shape[0] * flat_x.shape[1]))
            runtime.record_execution(f'gpu_native_train:{backward_kind}', input_name='grad_activation_output', output_name='grad_norm_output', node_count=1)
        grad_norm_host = runtime.stage_to_host(grad_activation_t)
        if use_native_layernorm:
            lib.layernorm_nd_backward(
                grad_activation_t.device_ptr,
                input_t.device_ptr,
                norm_weight_t.device_ptr,
                grad_input_t.device_ptr,
                grad_norm_weight_t.device_ptr,
                grad_norm_bias_t.device_ptr,
                int(flat_x.shape[0]),
                int(flat_x.shape[1]),
                float(norm_eps),
            )
            runtime.record_execution(
                'gpu_native_train:layernorm_backward',
                input_name='grad_norm_output',
                output_name='grad_input',
                node_count=1,
            )
            grad_input_flat = runtime.stage_to_host(grad_input_t)
        else:
            mean = flat_x.mean(axis=1, keepdims=True).astype(np.float32)
            var = flat_x.var(axis=1, keepdims=True).astype(np.float32)
            inv_std = (1.0 / np.sqrt(var + norm_eps)).astype(np.float32)
            x_hat = ((flat_x - mean) * inv_std).astype(np.float32)
            grad_norm_weight = (grad_norm_host * x_hat).sum(axis=0).astype(np.float32)
            grad_norm_bias = grad_norm_host.sum(axis=0).astype(np.float32)
            dxhat = (grad_norm_host * norm_weight_f32.reshape(1, -1)).astype(np.float32)
            sum_dxhat = dxhat.sum(axis=1, keepdims=True).astype(np.float32)
            sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True).astype(np.float32)
            grad_input_flat = (
                (inv_std / float(flat_x.shape[1]))
                * (float(flat_x.shape[1]) * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
            ).astype(np.float32)
            runtime.record_execution(
                'gpu_native_train:layernorm_backward_reference',
                input_name='grad_norm_output',
                output_name='grad_input',
                node_count=1,
            )
            np.copyto(grad_norm_weight_t.data, grad_norm_weight)
            np.copyto(grad_norm_bias_t.data, grad_norm_bias)
            runtime.sync_tensor_to_device(grad_norm_weight_t)
            runtime.sync_tensor_to_device(grad_norm_bias_t)

        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_norm_weight', output_name='norm_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-layernorm-linear-training-step')
        result_norm_output = runtime.stage_to_host(norm_t) if activation_key is None else runtime.stage_to_host(activated_t)
        return NativeGpuLayerNormLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            norm_output=result_norm_output,
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_norm_output=grad_norm_host,
            grad_input=grad_input_flat.reshape(x_f32.shape),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(flat_x.shape[0]),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_layernorm2d_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    norm_eps: float = 1e-6,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuLayerNorm2dLinearTrainingStepResult:
    """Run one native GPU LayerNorm2d + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_layernorm2d_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    flat_features = channels * height * width
    if norm_weight_f32.shape != (channels,) or norm_bias_f32.shape != (channels,):
        raise ValueError('native_gpu_layernorm2d_linear_training_step expects norm weight/bias with shape (C,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_layernorm2d_linear_training_step expects linear_weight with shape (out_f, C*H*W), '
            f'got linear_weight={linear_w_f32.shape} for flattened_features={flat_features}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_layernorm2d_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_layernorm2d_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_layernorm2d_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
    tensors = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    input_t = stage(x_f32, 'input')
    labels_t = stage(labels_i32, 'labels')
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    norm_t = alloc((n, channels, height, width), 'norm_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_norm_t = alloc((n, channels, height, width), 'grad_norm_output')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_norm_weight_t = alloc((channels,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((channels,), 'grad_norm_bias')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.layernorm2d_forward(input_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, channels, height, width, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_forward', input_name='input', output_name='norm_output', node_count=1)
        lib.dense_forward(norm_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='norm_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, norm_t.device_ptr, linear_w_t.device_ptr, grad_norm_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.layernorm2d_backward(grad_norm_t.device_ptr, input_t.device_ptr, norm_weight_t.device_ptr, grad_input_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, channels, height, width, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_backward', input_name='grad_norm_output', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_norm_weight', output_name='norm_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-layernorm2d-linear-training-step')
        return NativeGpuLayerNorm2dLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            norm_output=runtime.stage_to_host(norm_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)

