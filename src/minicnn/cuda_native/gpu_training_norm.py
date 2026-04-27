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
    NativeGpuBatchNormLinearTrainingStepResult,
    NativeGpuGroupNormLinearTrainingStepResult,
)
from minicnn.cuda_native.gpu_training_layernorm import (
    native_gpu_layernorm2d_linear_training_step,
    native_gpu_layernorm_linear_training_step,
)


def native_gpu_groupnorm_linear_training_step(
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
    num_groups: int = 1,
    norm_eps: float = 1e-5,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    return_intermediates: bool = True,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuGroupNormLinearTrainingStepResult:
    """Run one native GPU GroupNorm + Linear + SoftmaxCE + SGD step."""

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
        raise ValueError(f'native_gpu_groupnorm_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    if int(num_groups) <= 0 or channels % int(num_groups) != 0:
        raise ValueError('native_gpu_groupnorm_linear_training_step requires num_groups to divide input channels.')
    flat_features = channels * height * width
    if norm_weight_f32.shape != (channels,) or norm_bias_f32.shape != (channels,):
        raise ValueError('native_gpu_groupnorm_linear_training_step expects norm weight/bias with shape (C,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_groupnorm_linear_training_step expects linear_weight with shape (out_f, C*H*W), '
            f'got linear_weight={linear_w_f32.shape} for flattened_features={flat_features}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_groupnorm_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_groupnorm_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_groupnorm_linear_training_step labels must be in [0, out_f)')

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
        lib.groupnorm_forward(input_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, channels, height, width, int(num_groups), float(norm_eps))
        runtime.record_execution('gpu_native_train:groupnorm_forward', input_name='input', output_name='norm_output', node_count=1)
        lib.dense_forward(norm_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='norm_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, norm_t.device_ptr, linear_w_t.device_ptr, grad_norm_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.groupnorm_backward(grad_norm_t.device_ptr, input_t.device_ptr, norm_weight_t.device_ptr, grad_input_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, channels, height, width, int(num_groups), float(norm_eps))
        runtime.record_execution('gpu_native_train:groupnorm_backward', input_name='grad_norm_output', output_name='grad_input', node_count=1)
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
        runtime.synchronize('gpu-native-groupnorm-linear-training-step')
        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            norm_output = runtime.stage_to_host(norm_t)
            grad_logits = runtime.stage_to_host(grad_logits_t)
            grad_norm_output = runtime.stage_to_host(grad_norm_t)
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_norm_weight = runtime.stage_to_host(grad_norm_weight_t)
            grad_norm_bias = runtime.stage_to_host(grad_norm_bias_t)
            grad_linear_weight = runtime.stage_to_host(grad_linear_w_t)
            grad_linear_bias = runtime.stage_to_host(grad_linear_b_t)
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            norm_output = empty
            grad_logits = empty
            grad_norm_output = empty
            grad_input = empty
            grad_norm_weight = empty
            grad_norm_bias = empty
            grad_linear_weight = empty
            grad_linear_bias = empty
        updated_norm_weight = runtime.stage_to_host(norm_weight_t)
        updated_norm_bias = runtime.stage_to_host(norm_bias_t)
        updated_linear_weight = runtime.stage_to_host(linear_w_t)
        updated_linear_bias = runtime.stage_to_host(linear_b_t)
        copy_velocity = return_intermediates or float(momentum) != 0.0
        updated_norm_weight_velocity = runtime.stage_to_host(norm_wv_t) if copy_velocity else None
        updated_norm_bias_velocity = runtime.stage_to_host(norm_bv_t) if copy_velocity else None
        updated_linear_weight_velocity = runtime.stage_to_host(linear_wv_t) if copy_velocity else None
        updated_linear_bias_velocity = runtime.stage_to_host(linear_bv_t) if copy_velocity else None
        return NativeGpuGroupNormLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            norm_output=norm_output,
            grad_logits=grad_logits,
            grad_norm_output=grad_norm_output,
            grad_input=grad_input,
            grad_norm_weight=grad_norm_weight,
            grad_norm_bias=grad_norm_bias,
            grad_linear_weight=grad_linear_weight,
            grad_linear_bias=grad_linear_bias,
            updated_norm_weight=updated_norm_weight,
            updated_norm_bias=updated_norm_bias,
            updated_linear_weight=updated_linear_weight,
            updated_linear_bias=updated_linear_bias,
            updated_norm_weight_velocity=updated_norm_weight_velocity,
            updated_norm_bias_velocity=updated_norm_bias_velocity,
            updated_linear_weight_velocity=updated_linear_weight_velocity,
            updated_linear_bias_velocity=updated_linear_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_batchnorm_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    bn_eps: float = 1e-5,
    bn_momentum: float = 0.1,
    bn_weight_velocity: np.ndarray | None = None,
    bn_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    return_intermediates: bool = True,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuBatchNormLinearTrainingStepResult:
    """Run one native GPU BatchNorm2d + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    bn_weight_f32 = np.ascontiguousarray(bn_weight, dtype=np.float32)
    bn_bias_f32 = np.ascontiguousarray(bn_bias, dtype=np.float32)
    running_mean_f32 = np.ascontiguousarray(running_mean, dtype=np.float32)
    running_var_f32 = np.ascontiguousarray(running_var, dtype=np.float32)
    linear_weight_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_bias_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    bn_weight_velocity_f32 = (
        np.zeros_like(bn_weight_f32)
        if bn_weight_velocity is None
        else np.ascontiguousarray(bn_weight_velocity, dtype=np.float32)
    )
    bn_bias_velocity_f32 = (
        np.zeros_like(bn_bias_f32)
        if bn_bias_velocity is None
        else np.ascontiguousarray(bn_bias_velocity, dtype=np.float32)
    )
    linear_weight_velocity_f32 = (
        np.zeros_like(linear_weight_f32)
        if linear_weight_velocity is None
        else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    )
    linear_bias_velocity_f32 = (
        np.zeros_like(linear_bias_f32)
        if linear_bias_velocity is None
        else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_batchnorm_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    flat_features = channels * height * width
    for label, value in (
        ('bn_weight', bn_weight_f32),
        ('bn_bias', bn_bias_f32),
        ('running_mean', running_mean_f32),
        ('running_var', running_var_f32),
    ):
        if value.shape != (channels,):
            raise ValueError(
                f'native_gpu_batchnorm_linear_training_step expects {label} with shape {(channels,)}, got {value.shape}'
            )
    if linear_weight_f32.ndim != 2 or linear_weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_batchnorm_linear_training_step expects linear_weight with shape (out_f, C*H*W), '
            f'got linear_weight={linear_weight_f32.shape} for flat_features={flat_features}'
        )
    if linear_bias_f32.shape != (linear_weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_batchnorm_linear_training_step expects linear_bias with shape (out_f,), '
            f'got linear_bias={linear_bias_f32.shape} for linear_weight={linear_weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_batchnorm_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_weight_f32.shape[0]):
        raise ValueError('native_gpu_batchnorm_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_weight_f32.shape[0])
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
    bn_weight_t = stage(bn_weight_f32, 'bn_weight')
    bn_bias_t = stage(bn_bias_f32, 'bn_bias')
    running_mean_t = stage(running_mean_f32, 'running_mean')
    running_var_t = stage(running_var_f32, 'running_var')
    linear_weight_t = stage(linear_weight_f32, 'linear_weight')
    linear_bias_t = stage(linear_bias_f32, 'linear_bias')
    bn_weight_velocity_t = stage(bn_weight_velocity_f32, 'bn_weight_velocity')
    bn_bias_velocity_t = stage(bn_bias_velocity_f32, 'bn_bias_velocity')
    linear_weight_velocity_t = stage(linear_weight_velocity_f32, 'linear_weight_velocity')
    linear_bias_velocity_t = stage(linear_bias_velocity_f32, 'linear_bias_velocity')
    bn_output_t = alloc((n, channels, height, width), 'bn_output')
    x_hat_t = alloc((n, channels, height, width), 'x_hat')
    batch_mean_t = alloc((channels,), 'batch_mean')
    batch_inv_std_t = alloc((channels,), 'batch_inv_std')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_bn_output_t = alloc((n, channels, height, width), 'grad_bn_output')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_bn_weight_t = alloc((channels,), 'grad_bn_weight')
    grad_bn_bias_t = alloc((channels,), 'grad_bn_bias')
    grad_linear_weight_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_bias_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.bn_train_forward(
            bn_output_t.device_ptr,
            input_t.device_ptr,
            x_hat_t.device_ptr,
            batch_mean_t.device_ptr,
            batch_inv_std_t.device_ptr,
            running_mean_t.device_ptr,
            running_var_t.device_ptr,
            bn_weight_t.device_ptr,
            bn_bias_t.device_ptr,
            n,
            channels,
            height,
            width,
            float(bn_eps),
            float(bn_momentum),
        )
        runtime.record_execution('gpu_native_train:bn_train_forward', input_name='input', output_name='bn_output', node_count=1)
        lib.dense_forward(bn_output_t.device_ptr, linear_weight_t.device_ptr, linear_bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='bn_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(
            runtime,
            lib,
            logits_t,
            labels_t,
            probs_t,
            grad_logits_t,
            loss_sum_t,
            correct_t,
            n,
            out_f,
            label_smoothing=float(label_smoothing),
        )
        runtime.record_execution(
            loss_kind,
            input_name='logits',
            output_name='grad_logits',
            node_count=1,
        )
        lib.dense_backward_full(
            grad_logits_t.device_ptr,
            bn_output_t.device_ptr,
            linear_weight_t.device_ptr,
            grad_bn_output_t.device_ptr,
            grad_linear_weight_t.device_ptr,
            grad_linear_bias_t.device_ptr,
            n,
            flat_features,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.bn_backward(
            grad_input_t.device_ptr,
            grad_bn_weight_t.device_ptr,
            grad_bn_bias_t.device_ptr,
            grad_bn_output_t.device_ptr,
            x_hat_t.device_ptr,
            bn_weight_t.device_ptr,
            batch_inv_std_t.device_ptr,
            n,
            channels,
            height,
            width,
        )
        runtime.record_execution('gpu_native_train:bn_backward', input_name='grad_bn_output', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_bn_weight_t, int(bn_weight_f32.size)),
                (grad_bn_bias_t, int(bn_bias_f32.size)),
                (grad_linear_weight_t, int(linear_weight_f32.size)),
                (grad_linear_bias_t, int(linear_bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            updates = (
                (bn_weight_t, grad_bn_weight_t, bn_weight_velocity_t, int(bn_weight_f32.size)),
                (bn_bias_t, grad_bn_bias_t, bn_bias_velocity_t, int(bn_bias_f32.size)),
                (linear_weight_t, grad_linear_weight_t, linear_weight_velocity_t, int(linear_weight_f32.size)),
                (linear_bias_t, grad_linear_bias_t, linear_bias_velocity_t, int(linear_bias_f32.size)),
            )
            for value_t, grad_t, velocity_t, size in updates:
                lib.sgd_update_fused(
                    value_t.device_ptr,
                    grad_t.device_ptr,
                    velocity_t.device_ptr,
                    float(lr),
                    float(momentum),
                    float(weight_decay),
                    0.0,
                    1.0,
                    size,
                )
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            updates = (
                (bn_weight_t, grad_bn_weight_t, int(bn_weight_f32.size)),
                (bn_bias_t, grad_bn_bias_t, int(bn_bias_f32.size)),
                (linear_weight_t, grad_linear_weight_t, int(linear_weight_f32.size)),
                (linear_bias_t, grad_linear_bias_t, int(linear_bias_f32.size)),
            )
            for value_t, grad_t, size in updates:
                lib.apply_sgd_update(value_t.device_ptr, grad_t.device_ptr, float(lr), size)
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_bn_weight', output_name='bn_weight', node_count=1)

        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            grad_logits = runtime.stage_to_host(grad_logits_t)
            bn_output = runtime.stage_to_host(bn_output_t)
            x_hat = runtime.stage_to_host(x_hat_t)
            batch_mean = runtime.stage_to_host(batch_mean_t)
            batch_inv_std = runtime.stage_to_host(batch_inv_std_t)
            grad_bn_output = runtime.stage_to_host(grad_bn_output_t)
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_bn_weight = runtime.stage_to_host(grad_bn_weight_t)
            grad_bn_bias = runtime.stage_to_host(grad_bn_bias_t)
            grad_linear_weight = runtime.stage_to_host(grad_linear_weight_t)
            grad_linear_bias = runtime.stage_to_host(grad_linear_bias_t)
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            grad_logits = empty
            bn_output = empty
            x_hat = empty
            batch_mean = empty
            batch_inv_std = empty
            grad_bn_output = empty
            grad_input = empty
            grad_bn_weight = empty
            grad_bn_bias = empty
            grad_linear_weight = empty
            grad_linear_bias = empty
        updated_bn_weight = runtime.stage_to_host(bn_weight_t)
        updated_bn_bias = runtime.stage_to_host(bn_bias_t)
        updated_running_mean = runtime.stage_to_host(running_mean_t)
        updated_running_var = runtime.stage_to_host(running_var_t)
        updated_linear_weight = runtime.stage_to_host(linear_weight_t)
        updated_linear_bias = runtime.stage_to_host(linear_bias_t)
        copy_velocity = return_intermediates or float(momentum) != 0.0
        updated_bn_weight_velocity = runtime.stage_to_host(bn_weight_velocity_t) if copy_velocity else None
        updated_bn_bias_velocity = runtime.stage_to_host(bn_bias_velocity_t) if copy_velocity else None
        updated_linear_weight_velocity = runtime.stage_to_host(linear_weight_velocity_t) if copy_velocity else None
        updated_linear_bias_velocity = runtime.stage_to_host(linear_bias_velocity_t) if copy_velocity else None
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-batchnorm-linear-training-step')
        return NativeGpuBatchNormLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            bn_output=bn_output,
            x_hat=x_hat,
            batch_mean=batch_mean,
            batch_inv_std=batch_inv_std,
            grad_bn_output=grad_bn_output,
            grad_input=grad_input,
            grad_bn_weight=grad_bn_weight,
            grad_bn_bias=grad_bn_bias,
            grad_linear_weight=grad_linear_weight,
            grad_linear_bias=grad_linear_bias,
            updated_bn_weight=updated_bn_weight,
            updated_bn_bias=updated_bn_bias,
            updated_running_mean=updated_running_mean,
            updated_running_var=updated_running_var,
            updated_linear_weight=updated_linear_weight,
            updated_linear_bias=updated_linear_bias,
            updated_bn_weight_velocity=updated_bn_weight_velocity,
            updated_bn_bias_velocity=updated_bn_bias_velocity,
            updated_linear_weight_velocity=updated_linear_weight_velocity,
            updated_linear_bias_velocity=updated_linear_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)
