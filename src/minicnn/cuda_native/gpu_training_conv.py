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
    NativeGpuConvLinearTrainingStepResult,
    NativeGpuTwoConvReluPoolLinearTrainingStepResult,
)


def native_gpu_conv_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    conv_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    conv_weight_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    apply_relu_activation: bool = False,
    activation_kind: str | None = None,
    activation_alpha: float = 0.01,
    apply_maxpool: bool = False,
    conv_kind: str = 'conv2d',
    bound_lib: Any | None = None,
    device_runtime: DeviceRuntime | None = None,
    persistent_device_state: bool = False,
    persistent_cache_prefix: str = 'conv_linear',
    return_intermediates: bool = True,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuConvLinearTrainingStepResult:
    """Run one native GPU conv-family + optional activation/pool + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    conv_w_f32 = np.ascontiguousarray(conv_weight, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    conv_wv_f32 = (
        np.zeros_like(conv_w_f32)
        if conv_weight_velocity is None
        else np.ascontiguousarray(conv_weight_velocity, dtype=np.float32)
    )
    linear_wv_f32 = (
        np.zeros_like(linear_w_f32)
        if linear_weight_velocity is None
        else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    )
    linear_bv_f32 = (
        np.zeros_like(linear_b_f32)
        if linear_bias_velocity is None
        else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_conv_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if conv_w_f32.ndim != 4:
        raise ValueError(
            'native_gpu_conv_linear_training_step expects conv_weight with shape (out_c, in_c_or_1, kh, kw), '
            f'got {conv_w_f32.shape}'
        )
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    out_c, conv_in_c, kh, kw = [int(v) for v in conv_w_f32.shape]
    normalized_conv_kind = str(conv_kind).lower()
    normalized_activation_kind = (
        str(activation_kind).strip().lower().replace('_', '')
        if activation_kind is not None
        else ('relu' if bool(apply_relu_activation) else None)
    )
    if normalized_conv_kind not in {'conv2d', 'depthwise'}:
        raise ValueError(f'native_gpu_conv_linear_training_step got unsupported conv_kind={conv_kind!r}')
    if normalized_activation_kind not in {None, 'relu', 'leakyrelu', 'sigmoid', 'tanh', 'silu', 'gelu'}:
        raise ValueError(
            'native_gpu_conv_linear_training_step got unsupported '
            f'activation_kind={activation_kind!r}'
        )
    if normalized_conv_kind == 'depthwise':
        if conv_in_c != 1:
            raise ValueError('native_gpu_conv_linear_training_step depthwise mode expects conv_weight shape (out_c, 1, kh, kw).')
        if out_c % in_c != 0:
            raise ValueError('native_gpu_conv_linear_training_step depthwise mode requires out_c to be a multiple of input channels.')
    elif conv_in_c != in_c:
        raise ValueError(f'conv_weight input channels {conv_in_c} do not match x channels {in_c}')
    out_h = height - kh + 1
    out_w = width - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError('native_gpu_conv_linear_training_step requires valid Conv2d output dimensions.')
    conv_features = out_c * out_h * out_w
    if bool(apply_maxpool):
        if out_h % 2 != 0 or out_w % 2 != 0:
            raise ValueError('native_gpu_conv_linear_training_step requires even Conv2d H/W before 2x2 MaxPool2d.')
        pool_h = out_h // 2
        pool_w = out_w // 2
        dense_features = out_c * pool_h * pool_w
    else:
        pool_h = out_h
        pool_w = out_w
        dense_features = conv_features
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != dense_features:
        raise ValueError(
            'native_gpu_conv_linear_training_step expects linear_weight with shape (classes, flattened_features), '
            f'got linear_weight={linear_w_f32.shape} for flattened_features={dense_features}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError(
            'native_gpu_conv_linear_training_step expects linear_bias with shape (classes,), '
            f'got {linear_b_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_conv_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_conv_linear_training_step labels must be in [0, classes)')

    lib = _load_bound_lib(bound_lib if bound_lib is not None else (device_runtime.bound_lib if device_runtime is not None else None))
    runtime = device_runtime if device_runtime is not None else DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    runtime.bound_lib = lib
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    classes = int(linear_w_f32.shape[0])
    patch_size = in_c * kh * kw
    spatial_size = n * out_h * out_w
    tensors = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def stage_state(array: np.ndarray, name: str):
        if persistent_device_state:
            tensor = runtime.stage_persistent_to_device(
                array,
                key=f'{persistent_cache_prefix}:{name}',
                name=name,
                update_on_reuse=False,
            )
        else:
            tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    input_t = stage(x_f32, 'input')
    labels_t = stage(labels_i32, 'labels')
    conv_w_t = stage_state(conv_w_f32, 'conv_weight')
    depthwise_bias_t = stage(np.zeros((out_c,), dtype=np.float32), 'depthwise_bias') if normalized_conv_kind == 'depthwise' else None
    linear_w_t = stage_state(linear_w_f32, 'linear_weight')
    linear_b_t = stage_state(linear_b_f32, 'linear_bias')
    conv_wv_t = stage_state(conv_wv_f32, 'conv_weight_velocity')
    linear_wv_t = stage_state(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage_state(linear_bv_f32, 'linear_bias_velocity')
    col_t = alloc((patch_size, spatial_size), 'conv_col') if normalized_conv_kind == 'conv2d' else None
    conv_raw_t = alloc((out_c, n, out_h, out_w), 'conv_raw_cnhw') if normalized_conv_kind == 'conv2d' else None
    conv_t = alloc((n, out_c, out_h, out_w), 'conv_output')
    activation_input_t = alloc((n, out_c, out_h, out_w), 'conv_activation_input') if normalized_activation_kind is not None else None
    pooled_t = alloc((n, out_c, pool_h, pool_w), 'pooled') if bool(apply_maxpool) else None
    logits_t = alloc((n, classes), 'logits')
    probs_t = alloc((n, classes), 'probs')
    grad_logits_t = alloc((n, classes), 'grad_logits')
    grad_conv_t = alloc((n, out_c, out_h, out_w), 'grad_conv_output')
    grad_pooled_t = alloc((n, out_c, pool_h, pool_w), 'grad_pooled') if bool(apply_maxpool) else None
    grad_conv_cnhw_t = alloc((out_c, n, out_h, out_w), 'grad_conv_cnhw') if normalized_conv_kind == 'conv2d' else None
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_conv_w_t = alloc(tuple(int(v) for v in conv_w_f32.shape), 'grad_conv_weight')
    grad_depthwise_bias_t = alloc((out_c,), 'grad_depthwise_bias') if normalized_conv_kind == 'depthwise' else None
    grad_linear_w_t = alloc((classes, dense_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((classes,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        if normalized_conv_kind == 'depthwise':
            assert depthwise_bias_t is not None
            lib.depthwise_conv2d_forward(
                input_t.device_ptr,
                conv_w_t.device_ptr,
                depthwise_bias_t.device_ptr,
                conv_t.device_ptr,
                n,
                in_c,
                height,
                width,
                out_c,
                kh,
                kw,
                out_h,
                out_w,
                1,
                1,
                0,
                0,
                0,
            )
            runtime.record_execution('gpu_native_train:depthwise_conv2d_forward', input_name='input', output_name='conv_output', node_count=1)
        else:
            assert col_t is not None
            assert conv_raw_t is not None
            lib.im2col_forward(input_t.device_ptr, col_t.device_ptr, n, in_c, height, width, kh, kw, out_h, out_w)
            lib.gemm_forward(conv_w_t.device_ptr, col_t.device_ptr, conv_raw_t.device_ptr, out_c, spatial_size, patch_size)
            lib.cnhw_to_nchw(conv_raw_t.device_ptr, conv_t.device_ptr, n, out_c, out_h, out_w)
            runtime.record_execution('gpu_native_train:conv2d_im2col_gemm', input_name='input', output_name='conv_output', node_count=1)
        activation_elements = int(n * conv_features)
        if activation_input_t is not None:
            lib.gpu_memcpy_d2d(activation_input_t.device_ptr, conv_t.device_ptr, conv_t.nbytes)
        if normalized_activation_kind == 'relu':
            lib.apply_relu(conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:apply_relu', input_name='conv_output', output_name='conv_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_forward(conv_t.device_ptr, float(activation_alpha), activation_elements)
            runtime.record_execution('gpu_native_train:leaky_relu_forward', input_name='conv_output', output_name='conv_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_forward(conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:sigmoid_forward', input_name='conv_output', output_name='conv_output', node_count=1)
        elif normalized_activation_kind == 'tanh':
            lib.tanh_forward(conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:tanh_forward', input_name='conv_output', output_name='conv_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_forward(conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:silu_forward', input_name='conv_output', output_name='conv_output', node_count=1)
        elif normalized_activation_kind == 'gelu':
            lib.gelu_forward(conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:gelu_forward', input_name='conv_output', output_name='conv_output', node_count=1)
        dense_input_t = conv_t
        dense_input_name = 'conv_output'
        if bool(apply_maxpool):
            assert pooled_t is not None
            lib.apply_maxpool(conv_t.device_ptr, pooled_t.device_ptr, n, out_c, out_h, out_w)
            runtime.record_execution('gpu_native_train:apply_maxpool', input_name='conv_output', output_name='pooled', node_count=1)
            dense_input_t = pooled_t
            dense_input_name = 'pooled'
        lib.dense_forward(dense_input_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, dense_features, classes)
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
            n,
            classes,
            label_smoothing=float(label_smoothing),
        )
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        dense_grad_t = grad_pooled_t if bool(apply_maxpool) else grad_conv_t
        assert dense_grad_t is not None
        lib.dense_backward_full(
            grad_logits_t.device_ptr,
            dense_input_t.device_ptr,
            linear_w_t.device_ptr,
            dense_grad_t.device_ptr,
            grad_linear_w_t.device_ptr,
            grad_linear_b_t.device_ptr,
            n,
            dense_features,
            classes,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        if bool(apply_maxpool):
            assert grad_pooled_t is not None
            lib.maxpool_backward_nchw(
                grad_pooled_t.device_ptr,
                conv_t.device_ptr,
                grad_conv_t.device_ptr,
                n,
                out_c,
                out_h,
                out_w,
                pool_h,
                pool_w,
            )
            runtime.record_execution('gpu_native_train:maxpool_backward_nchw', input_name='grad_pooled', output_name='grad_conv_output', node_count=1)
        activation_grad_input_t = conv_t if activation_input_t is None else activation_input_t
        if normalized_activation_kind == 'relu':
            lib.apply_relu_backward(activation_grad_input_t.device_ptr, grad_conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:apply_relu_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_backward(activation_grad_input_t.device_ptr, grad_conv_t.device_ptr, float(activation_alpha), activation_elements)
            runtime.record_execution('gpu_native_train:leaky_relu_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_backward(activation_grad_input_t.device_ptr, grad_conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:sigmoid_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        elif normalized_activation_kind == 'tanh':
            lib.tanh_backward(activation_grad_input_t.device_ptr, grad_conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:tanh_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_backward(activation_grad_input_t.device_ptr, grad_conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:silu_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        elif normalized_activation_kind == 'gelu':
            lib.gelu_backward(activation_grad_input_t.device_ptr, grad_conv_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:gelu_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        if normalized_conv_kind == 'depthwise':
            assert grad_depthwise_bias_t is not None
            lib.depthwise_conv2d_backward(
                grad_conv_t.device_ptr,
                input_t.device_ptr,
                conv_w_t.device_ptr,
                grad_input_t.device_ptr,
                grad_conv_w_t.device_ptr,
                grad_depthwise_bias_t.device_ptr,
                n,
                in_c,
                height,
                width,
                out_c,
                kh,
                kw,
                out_h,
                out_w,
                1,
                1,
                0,
                0,
                0,
            )
            runtime.record_execution('gpu_native_train:depthwise_conv2d_backward', input_name='grad_conv_output', output_name='grad_conv_weight', node_count=1)
        else:
            assert grad_conv_cnhw_t is not None
            lib.nchw_to_cnhw(grad_conv_t.device_ptr, grad_conv_cnhw_t.device_ptr, n, out_c, out_h, out_w)
            lib.conv_backward(
                grad_conv_cnhw_t.device_ptr,
                input_t.device_ptr,
                conv_w_t.device_ptr,
                grad_conv_w_t.device_ptr,
                grad_input_t.device_ptr,
                n,
                in_c,
                height,
                width,
                kh,
                kw,
                out_h,
                out_w,
                out_c,
            )
            runtime.record_execution('gpu_native_train:conv_backward', input_name='grad_conv_output', output_name='grad_conv_weight', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_conv_w_t, int(conv_w_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(conv_w_t.device_ptr, grad_conv_w_t.device_ptr, conv_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(conv_w_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(conv_w_t.device_ptr, grad_conv_w_t.device_ptr, float(lr), int(conv_w_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_conv_weight', output_name='conv_weight', node_count=1)

        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            conv_output = runtime.stage_to_host(conv_t)
            pooled_output = runtime.stage_to_host(pooled_t) if pooled_t is not None else None
            grad_logits = runtime.stage_to_host(grad_logits_t)
            grad_conv_output = runtime.stage_to_host(grad_conv_t)
            grad_pooled = runtime.stage_to_host(grad_pooled_t) if grad_pooled_t is not None else None
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_conv_weight = runtime.stage_to_host(grad_conv_w_t)
            grad_linear_weight = runtime.stage_to_host(grad_linear_w_t)
            grad_linear_bias = runtime.stage_to_host(grad_linear_b_t)
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            conv_output = empty
            pooled_output = empty if pooled_t is not None else None
            grad_logits = empty
            grad_conv_output = empty
            grad_pooled = empty if grad_pooled_t is not None else None
            grad_input = empty
            grad_conv_weight = empty
            grad_linear_weight = empty
            grad_linear_bias = empty
        updated_conv_weight = runtime.stage_to_host(conv_w_t)
        updated_linear_weight = runtime.stage_to_host(linear_w_t)
        updated_linear_bias = runtime.stage_to_host(linear_b_t)
        copy_velocity = return_intermediates or float(momentum) != 0.0
        updated_conv_weight_velocity = runtime.stage_to_host(conv_wv_t) if copy_velocity else None
        updated_linear_weight_velocity = runtime.stage_to_host(linear_wv_t) if copy_velocity else None
        updated_linear_bias_velocity = runtime.stage_to_host(linear_bv_t) if copy_velocity else None
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-conv-linear-training-step')
        return NativeGpuConvLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            conv_output=conv_output,
            grad_logits=grad_logits,
            grad_conv_output=grad_conv_output,
            grad_input=grad_input,
            grad_conv_weight=grad_conv_weight,
            grad_linear_weight=grad_linear_weight,
            grad_linear_bias=grad_linear_bias,
            updated_conv_weight=updated_conv_weight,
            updated_linear_weight=updated_linear_weight,
            updated_linear_bias=updated_linear_bias,
            updated_conv_weight_velocity=updated_conv_weight_velocity,
            updated_linear_weight_velocity=updated_linear_weight_velocity,
            updated_linear_bias_velocity=updated_linear_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
            pooled_output=pooled_output,
            grad_pooled=grad_pooled,
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_two_conv_relu_pool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    conv1_weight: np.ndarray,
    conv2_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    conv1_weight_velocity: np.ndarray | None = None,
    conv2_weight_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    activation_kind: str | None = None,
    activation_alpha: float = 0.01,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
    return_intermediates: bool = True,
) -> NativeGpuTwoConvReluPoolLinearTrainingStepResult:
    """Run one native GPU Conv/activation/Conv/activation/MaxPool/Linear training step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    conv1_w_f32 = np.ascontiguousarray(conv1_weight, dtype=np.float32)
    conv2_w_f32 = np.ascontiguousarray(conv2_weight, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    conv1_wv_f32 = np.zeros_like(conv1_w_f32) if conv1_weight_velocity is None else np.ascontiguousarray(conv1_weight_velocity, dtype=np.float32)
    conv2_wv_f32 = np.zeros_like(conv2_w_f32) if conv2_weight_velocity is None else np.ascontiguousarray(conv2_weight_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)

    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_two_conv_relu_pool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if conv1_w_f32.ndim != 4 or conv2_w_f32.ndim != 4:
        raise ValueError('native_gpu_two_conv_relu_pool_linear_training_step expects conv weights with shape (out_c, in_c, kh, kw).')
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    conv1_out_c, conv1_in_c, k1h, k1w = [int(v) for v in conv1_w_f32.shape]
    conv2_out_c, conv2_in_c, k2h, k2w = [int(v) for v in conv2_w_f32.shape]
    normalized_activation_kind = str(activation_kind or 'ReLU').strip().lower().replace('_', '')
    if conv1_in_c != in_c:
        raise ValueError(f'conv1 input channels {conv1_in_c} do not match x channels {in_c}')
    if conv2_in_c != conv1_out_c:
        raise ValueError(f'conv2 input channels {conv2_in_c} do not match conv1 output channels {conv1_out_c}')
    if normalized_activation_kind not in {'relu', 'leakyrelu', 'sigmoid', 'tanh', 'silu', 'gelu'}:
        raise ValueError(
            'native_gpu_two_conv_relu_pool_linear_training_step got unsupported '
            f'activation_kind={activation_kind!r}'
        )
    conv1_h = height - k1h + 1
    conv1_w = width - k1w + 1
    conv2_h = conv1_h - k2h + 1
    conv2_w = conv1_w - k2w + 1
    if conv1_h <= 0 or conv1_w <= 0 or conv2_h <= 0 or conv2_w <= 0:
        raise ValueError('native_gpu_two_conv_relu_pool_linear_training_step requires valid Conv2d output dimensions.')
    if conv2_h % 2 != 0 or conv2_w % 2 != 0:
        raise ValueError('native_gpu_two_conv_relu_pool_linear_training_step requires even Conv2d H/W before 2x2 MaxPool2d.')
    pool_h = conv2_h // 2
    pool_w = conv2_w // 2
    dense_features = conv2_out_c * pool_h * pool_w
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != dense_features:
        raise ValueError(
            'native_gpu_two_conv_relu_pool_linear_training_step expects linear_weight with shape '
            f'(classes, {dense_features}), got {linear_w_f32.shape}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_two_conv_relu_pool_linear_training_step expects linear_bias with shape (classes,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_two_conv_relu_pool_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_two_conv_relu_pool_linear_training_step labels must be in [0, classes)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    classes = int(linear_w_f32.shape[0])
    conv1_patch = in_c * k1h * k1w
    conv2_patch = conv1_out_c * k2h * k2w
    conv1_spatial = n * conv1_h * conv1_w
    conv2_spatial = n * conv2_h * conv2_w
    conv1_features = conv1_out_c * conv1_h * conv1_w
    conv2_features = conv2_out_c * conv2_h * conv2_w
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
    conv1_w_t = stage(conv1_w_f32, 'conv1_weight')
    conv2_w_t = stage(conv2_w_f32, 'conv2_weight')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    conv1_wv_t = stage(conv1_wv_f32, 'conv1_weight_velocity')
    conv2_wv_t = stage(conv2_wv_f32, 'conv2_weight_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    conv1_col_t = alloc((conv1_patch, conv1_spatial), 'conv1_col')
    conv1_raw_t = alloc((conv1_out_c, n, conv1_h, conv1_w), 'conv1_raw_cnhw')
    conv1_t = alloc((n, conv1_out_c, conv1_h, conv1_w), 'conv1_output')
    conv1_activation_input_t = alloc((n, conv1_out_c, conv1_h, conv1_w), 'conv1_activation_input')
    conv2_col_t = alloc((conv2_patch, conv2_spatial), 'conv2_col')
    conv2_raw_t = alloc((conv2_out_c, n, conv2_h, conv2_w), 'conv2_raw_cnhw')
    conv2_t = alloc((n, conv2_out_c, conv2_h, conv2_w), 'conv2_output')
    conv2_activation_input_t = alloc((n, conv2_out_c, conv2_h, conv2_w), 'conv2_activation_input')
    pooled_t = alloc((n, conv2_out_c, pool_h, pool_w), 'pooled')
    logits_t = alloc((n, classes), 'logits')
    probs_t = alloc((n, classes), 'probs')
    grad_logits_t = alloc((n, classes), 'grad_logits')
    grad_pooled_t = alloc((n, conv2_out_c, pool_h, pool_w), 'grad_pooled')
    grad_conv2_t = alloc((n, conv2_out_c, conv2_h, conv2_w), 'grad_conv2_output')
    grad_conv2_cnhw_t = alloc((conv2_out_c, n, conv2_h, conv2_w), 'grad_conv2_cnhw')
    grad_conv1_t = alloc((n, conv1_out_c, conv1_h, conv1_w), 'grad_conv1_output')
    grad_conv1_cnhw_t = alloc((conv1_out_c, n, conv1_h, conv1_w), 'grad_conv1_cnhw')
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_conv1_w_t = alloc((conv1_out_c, in_c, k1h, k1w), 'grad_conv1_weight')
    grad_conv2_w_t = alloc((conv2_out_c, conv1_out_c, k2h, k2w), 'grad_conv2_weight')
    grad_linear_w_t = alloc((classes, dense_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((classes,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.im2col_forward(input_t.device_ptr, conv1_col_t.device_ptr, n, in_c, height, width, k1h, k1w, conv1_h, conv1_w)
        lib.gemm_forward(conv1_w_t.device_ptr, conv1_col_t.device_ptr, conv1_raw_t.device_ptr, conv1_out_c, conv1_spatial, conv1_patch)
        lib.cnhw_to_nchw(conv1_raw_t.device_ptr, conv1_t.device_ptr, n, conv1_out_c, conv1_h, conv1_w)
        runtime.record_execution('gpu_native_train:conv2d_1_im2col_gemm', input_name='input', output_name='conv1_output', node_count=1)
        lib.gpu_memcpy_d2d(conv1_activation_input_t.device_ptr, conv1_t.device_ptr, conv1_t.nbytes)
        if normalized_activation_kind == 'relu':
            lib.apply_relu(conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:apply_relu_1', input_name='conv1_output', output_name='conv1_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_forward(conv1_t.device_ptr, float(activation_alpha), int(n * conv1_features))
            runtime.record_execution('gpu_native_train:leaky_relu_forward_1', input_name='conv1_output', output_name='conv1_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_forward(conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:sigmoid_forward_1', input_name='conv1_output', output_name='conv1_output', node_count=1)
        elif normalized_activation_kind == 'tanh':
            lib.tanh_forward(conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:tanh_forward_1', input_name='conv1_output', output_name='conv1_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_forward(conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:silu_forward_1', input_name='conv1_output', output_name='conv1_output', node_count=1)
        else:
            lib.gelu_forward(conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:gelu_forward_1', input_name='conv1_output', output_name='conv1_output', node_count=1)
        lib.im2col_forward(conv1_t.device_ptr, conv2_col_t.device_ptr, n, conv1_out_c, conv1_h, conv1_w, k2h, k2w, conv2_h, conv2_w)
        lib.gemm_forward(conv2_w_t.device_ptr, conv2_col_t.device_ptr, conv2_raw_t.device_ptr, conv2_out_c, conv2_spatial, conv2_patch)
        lib.cnhw_to_nchw(conv2_raw_t.device_ptr, conv2_t.device_ptr, n, conv2_out_c, conv2_h, conv2_w)
        runtime.record_execution('gpu_native_train:conv2d_2_im2col_gemm', input_name='conv1_output', output_name='conv2_output', node_count=1)
        lib.gpu_memcpy_d2d(conv2_activation_input_t.device_ptr, conv2_t.device_ptr, conv2_t.nbytes)
        if normalized_activation_kind == 'relu':
            lib.apply_relu(conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:apply_relu_2', input_name='conv2_output', output_name='conv2_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_forward(conv2_t.device_ptr, float(activation_alpha), int(n * conv2_features))
            runtime.record_execution('gpu_native_train:leaky_relu_forward_2', input_name='conv2_output', output_name='conv2_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_forward(conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:sigmoid_forward_2', input_name='conv2_output', output_name='conv2_output', node_count=1)
        elif normalized_activation_kind == 'tanh':
            lib.tanh_forward(conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:tanh_forward_2', input_name='conv2_output', output_name='conv2_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_forward(conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:silu_forward_2', input_name='conv2_output', output_name='conv2_output', node_count=1)
        else:
            lib.gelu_forward(conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:gelu_forward_2', input_name='conv2_output', output_name='conv2_output', node_count=1)
        lib.apply_maxpool(conv2_t.device_ptr, pooled_t.device_ptr, n, conv2_out_c, conv2_h, conv2_w)
        runtime.record_execution('gpu_native_train:apply_maxpool', input_name='conv2_output', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, dense_features, classes)
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
            classes,
            label_smoothing=float(label_smoothing),
        )
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(
            grad_logits_t.device_ptr,
            pooled_t.device_ptr,
            linear_w_t.device_ptr,
            grad_pooled_t.device_ptr,
            grad_linear_w_t.device_ptr,
            grad_linear_b_t.device_ptr,
            n,
            dense_features,
            classes,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.maxpool_backward_nchw(
            grad_pooled_t.device_ptr,
            conv2_t.device_ptr,
            grad_conv2_t.device_ptr,
            n,
            conv2_out_c,
            conv2_h,
            conv2_w,
            pool_h,
            pool_w,
        )
        runtime.record_execution('gpu_native_train:maxpool_backward_nchw', input_name='grad_pooled', output_name='grad_conv2_output', node_count=1)
        if normalized_activation_kind == 'relu':
            lib.apply_relu_backward(conv2_activation_input_t.device_ptr, grad_conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:apply_relu_backward_2', input_name='conv2_output', output_name='grad_conv2_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_backward(conv2_activation_input_t.device_ptr, grad_conv2_t.device_ptr, float(activation_alpha), int(n * conv2_features))
            runtime.record_execution('gpu_native_train:leaky_relu_backward_2', input_name='conv2_output', output_name='grad_conv2_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_backward(conv2_activation_input_t.device_ptr, grad_conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:sigmoid_backward_2', input_name='conv2_output', output_name='grad_conv2_output', node_count=1)
        elif normalized_activation_kind == 'tanh':
            lib.tanh_backward(conv2_activation_input_t.device_ptr, grad_conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:tanh_backward_2', input_name='conv2_output', output_name='grad_conv2_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_backward(conv2_activation_input_t.device_ptr, grad_conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:silu_backward_2', input_name='conv2_output', output_name='grad_conv2_output', node_count=1)
        else:
            lib.gelu_backward(conv2_activation_input_t.device_ptr, grad_conv2_t.device_ptr, int(n * conv2_features))
            runtime.record_execution('gpu_native_train:gelu_backward_2', input_name='conv2_output', output_name='grad_conv2_output', node_count=1)
        lib.nchw_to_cnhw(grad_conv2_t.device_ptr, grad_conv2_cnhw_t.device_ptr, n, conv2_out_c, conv2_h, conv2_w)
        lib.conv_backward(
            grad_conv2_cnhw_t.device_ptr,
            conv1_t.device_ptr,
            conv2_w_t.device_ptr,
            grad_conv2_w_t.device_ptr,
            grad_conv1_t.device_ptr,
            n,
            conv1_out_c,
            conv1_h,
            conv1_w,
            k2h,
            k2w,
            conv2_h,
            conv2_w,
            conv2_out_c,
        )
        runtime.record_execution('gpu_native_train:conv_backward_2', input_name='grad_conv2_output', output_name='grad_conv2_weight', node_count=1)
        if normalized_activation_kind == 'relu':
            lib.apply_relu_backward(conv1_activation_input_t.device_ptr, grad_conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:apply_relu_backward_1', input_name='conv1_output', output_name='grad_conv1_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_backward(conv1_activation_input_t.device_ptr, grad_conv1_t.device_ptr, float(activation_alpha), int(n * conv1_features))
            runtime.record_execution('gpu_native_train:leaky_relu_backward_1', input_name='conv1_output', output_name='grad_conv1_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_backward(conv1_activation_input_t.device_ptr, grad_conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:sigmoid_backward_1', input_name='conv1_output', output_name='grad_conv1_output', node_count=1)
        elif normalized_activation_kind == 'tanh':
            lib.tanh_backward(conv1_activation_input_t.device_ptr, grad_conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:tanh_backward_1', input_name='conv1_output', output_name='grad_conv1_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_backward(conv1_activation_input_t.device_ptr, grad_conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:silu_backward_1', input_name='conv1_output', output_name='grad_conv1_output', node_count=1)
        else:
            lib.gelu_backward(conv1_activation_input_t.device_ptr, grad_conv1_t.device_ptr, int(n * conv1_features))
            runtime.record_execution('gpu_native_train:gelu_backward_1', input_name='conv1_output', output_name='grad_conv1_output', node_count=1)
        lib.nchw_to_cnhw(grad_conv1_t.device_ptr, grad_conv1_cnhw_t.device_ptr, n, conv1_out_c, conv1_h, conv1_w)
        lib.conv_backward(
            grad_conv1_cnhw_t.device_ptr,
            input_t.device_ptr,
            conv1_w_t.device_ptr,
            grad_conv1_w_t.device_ptr,
            grad_input_t.device_ptr,
            n,
            in_c,
            height,
            width,
            k1h,
            k1w,
            conv1_h,
            conv1_w,
            conv1_out_c,
        )
        runtime.record_execution('gpu_native_train:conv_backward_1', input_name='grad_conv1_output', output_name='grad_conv1_weight', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_conv1_w_t, int(conv1_w_f32.size)),
                (grad_conv2_w_t, int(conv2_w_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            updates = (
                (conv1_w_t, grad_conv1_w_t, conv1_wv_t, int(conv1_w_f32.size)),
                (conv2_w_t, grad_conv2_w_t, conv2_wv_t, int(conv2_w_f32.size)),
                (linear_w_t, grad_linear_w_t, linear_wv_t, int(linear_w_f32.size)),
                (linear_b_t, grad_linear_b_t, linear_bv_t, int(linear_b_f32.size)),
            )
            for value_t, grad_t, velocity_t, size in updates:
                lib.sgd_update_fused(value_t.device_ptr, grad_t.device_ptr, velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, size)
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            updates = (
                (conv1_w_t, grad_conv1_w_t, int(conv1_w_f32.size)),
                (conv2_w_t, grad_conv2_w_t, int(conv2_w_f32.size)),
                (linear_w_t, grad_linear_w_t, int(linear_w_f32.size)),
                (linear_b_t, grad_linear_b_t, int(linear_b_f32.size)),
            )
            for value_t, grad_t, size in updates:
                lib.apply_sgd_update(value_t.device_ptr, grad_t.device_ptr, float(lr), size)
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_conv1_weight', output_name='conv1_weight', node_count=1)

        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            conv1_output = runtime.stage_to_host(conv1_t)
            conv2_output = runtime.stage_to_host(conv2_t)
            pooled_output = runtime.stage_to_host(pooled_t)
            grad_logits = runtime.stage_to_host(grad_logits_t)
            grad_pooled = runtime.stage_to_host(grad_pooled_t)
            grad_conv2_output = runtime.stage_to_host(grad_conv2_t)
            grad_conv1_output = runtime.stage_to_host(grad_conv1_t)
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_conv1_weight = runtime.stage_to_host(grad_conv1_w_t)
            grad_conv2_weight = runtime.stage_to_host(grad_conv2_w_t)
            grad_linear_weight = runtime.stage_to_host(grad_linear_w_t)
            grad_linear_bias = runtime.stage_to_host(grad_linear_b_t)
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            conv1_output = empty
            conv2_output = empty
            pooled_output = empty
            grad_logits = empty
            grad_pooled = empty
            grad_conv2_output = empty
            grad_conv1_output = empty
            grad_input = empty
            grad_conv1_weight = empty
            grad_conv2_weight = empty
            grad_linear_weight = empty
            grad_linear_bias = empty
        updated_conv1_weight = runtime.stage_to_host(conv1_w_t)
        updated_conv2_weight = runtime.stage_to_host(conv2_w_t)
        updated_linear_weight = runtime.stage_to_host(linear_w_t)
        updated_linear_bias = runtime.stage_to_host(linear_b_t)
        copy_velocity = return_intermediates or float(momentum) != 0.0
        updated_conv1_weight_velocity = runtime.stage_to_host(conv1_wv_t) if copy_velocity else None
        updated_conv2_weight_velocity = runtime.stage_to_host(conv2_wv_t) if copy_velocity else None
        updated_linear_weight_velocity = runtime.stage_to_host(linear_wv_t) if copy_velocity else None
        updated_linear_bias_velocity = runtime.stage_to_host(linear_bv_t) if copy_velocity else None
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-two-conv-relu-pool-linear-training-step')
        return NativeGpuTwoConvReluPoolLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            conv1_output=conv1_output,
            conv2_output=conv2_output,
            pooled_output=pooled_output,
            grad_logits=grad_logits,
            grad_pooled=grad_pooled,
            grad_conv2_output=grad_conv2_output,
            grad_conv1_output=grad_conv1_output,
            grad_input=grad_input,
            grad_conv1_weight=grad_conv1_weight,
            grad_conv2_weight=grad_conv2_weight,
            grad_linear_weight=grad_linear_weight,
            grad_linear_bias=grad_linear_bias,
            updated_conv1_weight=updated_conv1_weight,
            updated_conv2_weight=updated_conv2_weight,
            updated_linear_weight=updated_linear_weight,
            updated_linear_bias=updated_linear_bias,
            updated_conv1_weight_velocity=updated_conv1_weight_velocity,
            updated_conv2_weight_velocity=updated_conv2_weight_velocity,
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
