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
    NativeGpuLinearTrainingStepResult,
    NativeGpuGroupNormLinearTrainingStepResult,
    NativeGpuLayerNorm2dLinearTrainingStepResult,
    NativeGpuPoolLinearTrainingStepResult,
    NativeGpuTwoLinearReluTrainingStepResult,
)


def native_gpu_pool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuPoolLinearTrainingStepResult:
    """Run one native GPU MaxPool2d(2,2) + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    weight_f32 = np.ascontiguousarray(weight, dtype=np.float32)
    bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
    weight_velocity_f32 = (
        np.zeros_like(weight_f32)
        if weight_velocity is None
        else np.ascontiguousarray(weight_velocity, dtype=np.float32)
    )
    bias_velocity_f32 = (
        np.zeros_like(bias_f32)
        if bias_velocity is None
        else np.ascontiguousarray(bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_pool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError('native_gpu_pool_linear_training_step requires even H/W for 2x2 stride-2 MaxPool2d.')
    pool_h = height // 2
    pool_w = width // 2
    flat_features = channels * pool_h * pool_w
    if weight_f32.ndim != 2 or weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_pool_linear_training_step expects weight with shape (out_f, C*H/2*W/2), '
            f'got weight={weight_f32.shape} for pooled_features={flat_features}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_pool_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_pool_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_pool_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(weight_f32.shape[0])
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
    weight_t = stage(weight_f32, 'weight')
    bias_t = stage(bias_f32, 'bias')
    weight_velocity_t = stage(weight_velocity_f32, 'weight_velocity')
    bias_velocity_t = stage(bias_velocity_f32, 'bias_velocity')
    pooled_t = alloc((n, channels, pool_h, pool_w), 'pooled')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pooled_t = alloc((n, channels, pool_h, pool_w), 'grad_pooled')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_weight_t = alloc((out_f, flat_features), 'grad_weight')
    grad_bias_t = alloc((out_f,), 'grad_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.apply_maxpool(input_t.device_ptr, pooled_t.device_ptr, n, channels, height, width)
        runtime.record_execution('gpu_native_train:apply_maxpool', input_name='input', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, weight_t.device_ptr, bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pooled', output_name='logits', node_count=1)
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
            pooled_t.device_ptr,
            weight_t.device_ptr,
            grad_pooled_t.device_ptr,
            grad_weight_t.device_ptr,
            grad_bias_t.device_ptr,
            n,
            flat_features,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_weight', node_count=1)
        lib.maxpool_backward_nchw(
            grad_pooled_t.device_ptr,
            input_t.device_ptr,
            grad_input_t.device_ptr,
            n,
            channels,
            height,
            width,
            pool_h,
            pool_w,
        )
        runtime.record_execution('gpu_native_train:maxpool_backward_nchw', input_name='grad_pooled', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_weight_t, int(weight_f32.size)),
                (grad_bias_t, int(bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(weight_f32.size))
            lib.sgd_update_fused(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(bias_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(weight_t.device_ptr, grad_weight_t.device_ptr, float(lr), int(weight_f32.size))
            lib.apply_sgd_update(bias_t.device_ptr, grad_bias_t.device_ptr, float(lr), int(bias_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_weight', output_name='weight', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        pooled = runtime.stage_to_host(pooled_t)
        grad_pooled = runtime.stage_to_host(grad_pooled_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_weight = runtime.stage_to_host(grad_weight_t)
        grad_bias = runtime.stage_to_host(grad_bias_t)
        updated_weight = runtime.stage_to_host(weight_t)
        updated_bias = runtime.stage_to_host(bias_t)
        updated_weight_velocity = runtime.stage_to_host(weight_velocity_t)
        updated_bias_velocity = runtime.stage_to_host(bias_velocity_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-pool-linear-training-step')
        return NativeGpuPoolLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            pooled=pooled,
            grad_pooled=grad_pooled,
            grad_input=grad_input,
            grad_weight=grad_weight,
            grad_bias=grad_bias,
            updated_weight=updated_weight,
            updated_bias=updated_bias,
            updated_weight_velocity=updated_weight_velocity,
            updated_bias_velocity=updated_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_avgpool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuPoolLinearTrainingStepResult:
    """Run one native GPU AvgPool2d(2,2) + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    weight_f32 = np.ascontiguousarray(weight, dtype=np.float32)
    bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
    weight_velocity_f32 = (
        np.zeros_like(weight_f32)
        if weight_velocity is None
        else np.ascontiguousarray(weight_velocity, dtype=np.float32)
    )
    bias_velocity_f32 = (
        np.zeros_like(bias_f32)
        if bias_velocity is None
        else np.ascontiguousarray(bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_avgpool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError('native_gpu_avgpool_linear_training_step requires even H/W for 2x2 stride-2 AvgPool2d.')
    pool_h = height // 2
    pool_w = width // 2
    flat_features = channels * pool_h * pool_w
    if weight_f32.ndim != 2 or weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_avgpool_linear_training_step expects weight with shape (out_f, C*H/2*W/2), '
            f'got weight={weight_f32.shape} for pooled_features={flat_features}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_avgpool_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_avgpool_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_avgpool_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(weight_f32.shape[0])
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
    weight_t = stage(weight_f32, 'weight')
    bias_t = stage(bias_f32, 'bias')
    weight_velocity_t = stage(weight_velocity_f32, 'weight_velocity')
    bias_velocity_t = stage(bias_velocity_f32, 'bias_velocity')
    pooled_t = alloc((n, channels, pool_h, pool_w), 'pooled')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pooled_t = alloc((n, channels, pool_h, pool_w), 'grad_pooled')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_weight_t = alloc((out_f, flat_features), 'grad_weight')
    grad_bias_t = alloc((out_f,), 'grad_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.avgpool2d_forward(input_t.device_ptr, pooled_t.device_ptr, n, channels, height, width, pool_h, pool_w, 2, 2, 2, 2, 0, 0)
        runtime.record_execution('gpu_native_train:avgpool2d_forward', input_name='input', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, weight_t.device_ptr, bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pooled', output_name='logits', node_count=1)
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
            pooled_t.device_ptr,
            weight_t.device_ptr,
            grad_pooled_t.device_ptr,
            grad_weight_t.device_ptr,
            grad_bias_t.device_ptr,
            n,
            flat_features,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_weight', node_count=1)
        lib.avgpool2d_backward(grad_pooled_t.device_ptr, grad_input_t.device_ptr, n, channels, height, width, pool_h, pool_w, 2, 2, 2, 2, 0, 0)
        runtime.record_execution('gpu_native_train:avgpool2d_backward', input_name='grad_pooled', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_weight_t, int(weight_f32.size)),
                (grad_bias_t, int(bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(weight_f32.size))
            lib.sgd_update_fused(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(bias_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(weight_t.device_ptr, grad_weight_t.device_ptr, float(lr), int(weight_f32.size))
            lib.apply_sgd_update(bias_t.device_ptr, grad_bias_t.device_ptr, float(lr), int(bias_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_weight', output_name='weight', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        pooled = runtime.stage_to_host(pooled_t)
        grad_pooled = runtime.stage_to_host(grad_pooled_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_weight = runtime.stage_to_host(grad_weight_t)
        grad_bias = runtime.stage_to_host(grad_bias_t)
        updated_weight = runtime.stage_to_host(weight_t)
        updated_bias = runtime.stage_to_host(bias_t)
        updated_weight_velocity = runtime.stage_to_host(weight_velocity_t)
        updated_bias_velocity = runtime.stage_to_host(bias_velocity_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-avgpool-linear-training-step')
        return NativeGpuPoolLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            pooled=pooled,
            grad_pooled=grad_pooled,
            grad_input=grad_input,
            grad_weight=grad_weight,
            grad_bias=grad_bias,
            updated_weight=updated_weight,
            updated_bias=updated_bias,
            updated_weight_velocity=updated_weight_velocity,
            updated_bias_velocity=updated_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_global_avgpool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuPoolLinearTrainingStepResult:
    """Run one native GPU GlobalAvgPool2d + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    weight_f32 = np.ascontiguousarray(weight, dtype=np.float32)
    bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
    weight_velocity_f32 = (
        np.zeros_like(weight_f32)
        if weight_velocity is None
        else np.ascontiguousarray(weight_velocity, dtype=np.float32)
    )
    bias_velocity_f32 = (
        np.zeros_like(bias_f32)
        if bias_velocity is None
        else np.ascontiguousarray(bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_global_avgpool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    flat_features = channels
    if weight_f32.ndim != 2 or weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_global_avgpool_linear_training_step expects weight with shape (out_f, C), '
            f'got weight={weight_f32.shape} for channels={flat_features}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_global_avgpool_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_global_avgpool_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_global_avgpool_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(weight_f32.shape[0])
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
    weight_t = stage(weight_f32, 'weight')
    bias_t = stage(bias_f32, 'bias')
    weight_velocity_t = stage(weight_velocity_f32, 'weight_velocity')
    bias_velocity_t = stage(bias_velocity_f32, 'bias_velocity')
    pooled_t = alloc((n, channels, 1, 1), 'pooled')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pooled_t = alloc((n, channels, 1, 1), 'grad_pooled')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_weight_t = alloc((out_f, flat_features), 'grad_weight')
    grad_bias_t = alloc((out_f,), 'grad_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.global_avgpool2d_forward(input_t.device_ptr, pooled_t.device_ptr, n, channels, height, width)
        runtime.record_execution('gpu_native_train:global_avgpool2d_forward', input_name='input', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, weight_t.device_ptr, bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pooled', output_name='logits', node_count=1)
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
            pooled_t.device_ptr,
            weight_t.device_ptr,
            grad_pooled_t.device_ptr,
            grad_weight_t.device_ptr,
            grad_bias_t.device_ptr,
            n,
            flat_features,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_weight', node_count=1)
        lib.global_avgpool2d_backward(grad_pooled_t.device_ptr, grad_input_t.device_ptr, n, channels, height, width)
        runtime.record_execution('gpu_native_train:global_avgpool2d_backward', input_name='grad_pooled', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_weight_t, int(weight_f32.size)),
                (grad_bias_t, int(bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(weight_f32.size))
            lib.sgd_update_fused(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(bias_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(weight_t.device_ptr, grad_weight_t.device_ptr, float(lr), int(weight_f32.size))
            lib.apply_sgd_update(bias_t.device_ptr, grad_bias_t.device_ptr, float(lr), int(bias_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_weight', output_name='weight', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        pooled = runtime.stage_to_host(pooled_t)
        grad_pooled = runtime.stage_to_host(grad_pooled_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_weight = runtime.stage_to_host(grad_weight_t)
        grad_bias = runtime.stage_to_host(grad_bias_t)
        updated_weight = runtime.stage_to_host(weight_t)
        updated_bias = runtime.stage_to_host(bias_t)
        updated_weight_velocity = runtime.stage_to_host(weight_velocity_t)
        updated_bias_velocity = runtime.stage_to_host(bias_velocity_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-global-avgpool-linear-training-step')
        return NativeGpuPoolLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            pooled=pooled,
            grad_pooled=grad_pooled,
            grad_input=grad_input,
            grad_weight=grad_weight,
            grad_bias=grad_bias,
            updated_weight=updated_weight,
            updated_bias=updated_bias,
            updated_weight_velocity=updated_weight_velocity,
            updated_bias_velocity=updated_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)
