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
    NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult,
)


def native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    depthwise_weight: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    pointwise1_weight: np.ndarray,
    pointwise2_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    norm_eps: float = 1e-6,
    depthwise_weight_velocity: np.ndarray | None = None,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    pointwise1_weight_velocity: np.ndarray | None = None,
    pointwise2_weight_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    activation_kind: str = 'GELU',
    activation_alpha: float = 0.01,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult:
    """Run native GPU DepthwiseConv2d + LayerNorm2d + PointwiseConv2d + activation + PointwiseConv2d + Linear."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    depthwise_w_f32 = np.ascontiguousarray(depthwise_weight, dtype=np.float32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    pointwise1_w_f32 = np.ascontiguousarray(pointwise1_weight, dtype=np.float32)
    pointwise2_w_f32 = np.ascontiguousarray(pointwise2_weight, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    depthwise_wv_f32 = np.zeros_like(depthwise_w_f32) if depthwise_weight_velocity is None else np.ascontiguousarray(depthwise_weight_velocity, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    pointwise1_wv_f32 = np.zeros_like(pointwise1_w_f32) if pointwise1_weight_velocity is None else np.ascontiguousarray(pointwise1_weight_velocity, dtype=np.float32)
    pointwise2_wv_f32 = np.zeros_like(pointwise2_w_f32) if pointwise2_weight_velocity is None else np.ascontiguousarray(pointwise2_weight_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    normalized_activation_kind = str(activation_kind).strip().lower().replace('_', '')

    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if depthwise_w_f32.ndim != 4 or depthwise_w_f32.shape[1] != 1:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects depthwise_weight shape (out_c, 1, kh, kw).')
    if pointwise1_w_f32.ndim != 4 or pointwise1_w_f32.shape[2:] != (1, 1):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects pointwise1_weight shape (out_c, in_c, 1, 1).')
    if pointwise2_w_f32.ndim != 4 or pointwise2_w_f32.shape[2:] != (1, 1):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects pointwise2_weight shape (out_c, in_c, 1, 1).')
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    depthwise_out_c, _depthwise_in_c, kh, kw = [int(v) for v in depthwise_w_f32.shape]
    hidden_c, pointwise1_in_c, _p1_kh, _p1_kw = [int(v) for v in pointwise1_w_f32.shape]
    pointwise_out_c, pointwise2_in_c, _p2_kh, _p2_kw = [int(v) for v in pointwise2_w_f32.shape]
    if depthwise_out_c % in_c != 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires depthwise out_c to be a multiple of input channels.')
    if pointwise1_in_c != depthwise_out_c:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires pointwise1 in_c to match depthwise out_c.')
    if pointwise2_in_c != hidden_c:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires pointwise2 in_c to match pointwise1 out_c.')
    out_h = height - kh + 1
    out_w = width - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires valid DepthwiseConv2d output dimensions.')
    flat_features = pointwise_out_c * out_h * out_w
    if norm_weight_f32.shape != (depthwise_out_c,) or norm_bias_f32.shape != (depthwise_out_c,):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects norm weight/bias with shape (depthwise_out_c,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects linear_weight with shape '
            f'(out_f, {flat_features}), got {linear_w_f32.shape}.'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step labels must be in [0, out_f).')
    if normalized_activation_kind not in {'gelu', 'leakyrelu', 'silu', 'sigmoid', 'tanh'}:
        raise ValueError(
            'native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step got unsupported '
            f'activation_kind={activation_kind!r}'
        )

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
    spatial_size = n * out_h * out_w
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
    depthwise_w_t = stage(depthwise_w_f32, 'depthwise_weight')
    depthwise_bias_t = stage(np.zeros((depthwise_out_c,), dtype=np.float32), 'depthwise_bias')
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    pointwise1_w_t = stage(pointwise1_w_f32, 'pointwise1_weight')
    pointwise2_w_t = stage(pointwise2_w_f32, 'pointwise2_weight')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    depthwise_wv_t = stage(depthwise_wv_f32, 'depthwise_weight_velocity')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    pointwise1_wv_t = stage(pointwise1_wv_f32, 'pointwise1_weight_velocity')
    pointwise2_wv_t = stage(pointwise2_wv_f32, 'pointwise2_weight_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    depthwise_t = alloc((n, depthwise_out_c, out_h, out_w), 'depthwise_output')
    norm_t = alloc((n, depthwise_out_c, out_h, out_w), 'norm_output')
    pw1_col_t = alloc((depthwise_out_c, spatial_size), 'pointwise1_col')
    pw1_raw_t = alloc((hidden_c, n, out_h, out_w), 'pointwise1_raw_cnhw')
    pw1_t = alloc((n, hidden_c, out_h, out_w), 'pointwise1_output')
    activation_t = alloc((n, hidden_c, out_h, out_w), 'activation_output')
    pw2_col_t = alloc((hidden_c, spatial_size), 'pointwise2_col')
    pw2_raw_t = alloc((pointwise_out_c, n, out_h, out_w), 'pointwise2_raw_cnhw')
    pw2_t = alloc((n, pointwise_out_c, out_h, out_w), 'pointwise2_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pw2_t = alloc((n, pointwise_out_c, out_h, out_w), 'grad_pointwise2_output')
    grad_pw2_cnhw_t = alloc((pointwise_out_c, n, out_h, out_w), 'grad_pointwise2_cnhw')
    grad_activation_t = alloc((n, hidden_c, out_h, out_w), 'grad_activation_output')
    grad_activation_before_gelu_t = alloc((n, hidden_c, out_h, out_w), 'grad_activation_before_gelu')
    grad_activation_cnhw_t = alloc((hidden_c, n, out_h, out_w), 'grad_activation_cnhw')
    grad_pw1_t = alloc((n, hidden_c, out_h, out_w), 'grad_pointwise1_output')
    grad_norm_t = alloc((n, depthwise_out_c, out_h, out_w), 'grad_norm_output')
    grad_depthwise_t = alloc((n, depthwise_out_c, out_h, out_w), 'grad_depthwise_output')
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_depthwise_w_t = alloc(tuple(int(v) for v in depthwise_w_f32.shape), 'grad_depthwise_weight')
    grad_depthwise_bias_t = alloc((depthwise_out_c,), 'grad_depthwise_bias')
    grad_norm_weight_t = alloc((depthwise_out_c,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((depthwise_out_c,), 'grad_norm_bias')
    grad_pw1_w_t = alloc(tuple(int(v) for v in pointwise1_w_f32.shape), 'grad_pointwise1_weight')
    grad_pw2_w_t = alloc(tuple(int(v) for v in pointwise2_w_f32.shape), 'grad_pointwise2_weight')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.depthwise_conv2d_forward(
            input_t.device_ptr,
            depthwise_w_t.device_ptr,
            depthwise_bias_t.device_ptr,
            depthwise_t.device_ptr,
            n,
            in_c,
            height,
            width,
            depthwise_out_c,
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
        runtime.record_execution('gpu_native_train:depthwise_conv2d_forward', input_name='input', output_name='depthwise_output', node_count=1)
        lib.layernorm2d_forward(depthwise_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, depthwise_out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_forward', input_name='depthwise_output', output_name='norm_output', node_count=1)
        lib.im2col_forward(norm_t.device_ptr, pw1_col_t.device_ptr, n, depthwise_out_c, out_h, out_w, 1, 1, out_h, out_w)
        lib.gemm_forward(pointwise1_w_t.device_ptr, pw1_col_t.device_ptr, pw1_raw_t.device_ptr, hidden_c, spatial_size, depthwise_out_c)
        lib.cnhw_to_nchw(pw1_raw_t.device_ptr, pw1_t.device_ptr, n, hidden_c, out_h, out_w)
        runtime.record_execution('gpu_native_train:pointwise1_conv2d_im2col_gemm', input_name='norm_output', output_name='pointwise1_output', node_count=1)
        activation_elements = int(n * hidden_c * out_h * out_w)
        lib.gpu_memcpy_d2d(activation_t.device_ptr, pw1_t.device_ptr, activation_t.nbytes)
        if normalized_activation_kind == 'gelu':
            lib.gelu_forward(activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:gelu_forward', input_name='pointwise1_output', output_name='activation_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_forward(activation_t.device_ptr, float(activation_alpha), activation_elements)
            runtime.record_execution('gpu_native_train:leaky_relu_forward', input_name='pointwise1_output', output_name='activation_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_forward(activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:silu_forward', input_name='pointwise1_output', output_name='activation_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_forward(activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:sigmoid_forward', input_name='pointwise1_output', output_name='activation_output', node_count=1)
        else:
            lib.tanh_forward(activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:tanh_forward', input_name='pointwise1_output', output_name='activation_output', node_count=1)
        lib.im2col_forward(activation_t.device_ptr, pw2_col_t.device_ptr, n, hidden_c, out_h, out_w, 1, 1, out_h, out_w)
        lib.gemm_forward(pointwise2_w_t.device_ptr, pw2_col_t.device_ptr, pw2_raw_t.device_ptr, pointwise_out_c, spatial_size, hidden_c)
        lib.cnhw_to_nchw(pw2_raw_t.device_ptr, pw2_t.device_ptr, n, pointwise_out_c, out_h, out_w)
        runtime.record_execution('gpu_native_train:pointwise2_conv2d_im2col_gemm', input_name='activation_output', output_name='pointwise2_output', node_count=1)
        lib.dense_forward(pw2_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pointwise2_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, pw2_t.device_ptr, linear_w_t.device_ptr, grad_pw2_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.nchw_to_cnhw(grad_pw2_t.device_ptr, grad_pw2_cnhw_t.device_ptr, n, pointwise_out_c, out_h, out_w)
        lib.conv_backward(
            grad_pw2_cnhw_t.device_ptr,
            activation_t.device_ptr,
            pointwise2_w_t.device_ptr,
            grad_pw2_w_t.device_ptr,
            grad_activation_t.device_ptr,
            n,
            hidden_c,
            out_h,
            out_w,
            1,
            1,
            out_h,
            out_w,
            pointwise_out_c,
        )
        runtime.record_execution('gpu_native_train:pointwise2_conv2d_backward', input_name='grad_pointwise2_output', output_name='grad_pointwise2_weight', node_count=1)
        lib.gpu_memcpy_d2d(grad_activation_before_gelu_t.device_ptr, grad_activation_t.device_ptr, grad_activation_before_gelu_t.nbytes)
        if normalized_activation_kind == 'gelu':
            lib.gelu_backward(pw1_t.device_ptr, grad_activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:gelu_backward', input_name='pointwise1_output', output_name='grad_pointwise1_output', node_count=1)
        elif normalized_activation_kind == 'leakyrelu':
            lib.leaky_relu_backward(pw1_t.device_ptr, grad_activation_t.device_ptr, float(activation_alpha), activation_elements)
            runtime.record_execution('gpu_native_train:leaky_relu_backward', input_name='pointwise1_output', output_name='grad_pointwise1_output', node_count=1)
        elif normalized_activation_kind == 'silu':
            lib.silu_backward(pw1_t.device_ptr, grad_activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:silu_backward', input_name='pointwise1_output', output_name='grad_pointwise1_output', node_count=1)
        elif normalized_activation_kind == 'sigmoid':
            lib.sigmoid_backward(pw1_t.device_ptr, grad_activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:sigmoid_backward', input_name='pointwise1_output', output_name='grad_pointwise1_output', node_count=1)
        else:
            lib.tanh_backward(pw1_t.device_ptr, grad_activation_t.device_ptr, activation_elements)
            runtime.record_execution('gpu_native_train:tanh_backward', input_name='pointwise1_output', output_name='grad_pointwise1_output', node_count=1)
        lib.nchw_to_cnhw(grad_activation_t.device_ptr, grad_activation_cnhw_t.device_ptr, n, hidden_c, out_h, out_w)
        lib.conv_backward(
            grad_activation_cnhw_t.device_ptr,
            norm_t.device_ptr,
            pointwise1_w_t.device_ptr,
            grad_pw1_w_t.device_ptr,
            grad_norm_t.device_ptr,
            n,
            depthwise_out_c,
            out_h,
            out_w,
            1,
            1,
            out_h,
            out_w,
            hidden_c,
        )
        runtime.record_execution('gpu_native_train:pointwise1_conv2d_backward', input_name='grad_pointwise1_output', output_name='grad_pointwise1_weight', node_count=1)
        lib.gpu_memcpy_d2d(grad_pw1_t.device_ptr, grad_activation_t.device_ptr, grad_pw1_t.nbytes)
        lib.layernorm2d_backward(grad_norm_t.device_ptr, depthwise_t.device_ptr, norm_weight_t.device_ptr, grad_depthwise_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, depthwise_out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_backward', input_name='grad_norm_output', output_name='grad_depthwise_output', node_count=1)
        lib.depthwise_conv2d_backward(
            grad_depthwise_t.device_ptr,
            input_t.device_ptr,
            depthwise_w_t.device_ptr,
            grad_input_t.device_ptr,
            grad_depthwise_w_t.device_ptr,
            grad_depthwise_bias_t.device_ptr,
            n,
            in_c,
            height,
            width,
            depthwise_out_c,
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
        runtime.record_execution('gpu_native_train:depthwise_conv2d_backward', input_name='grad_depthwise_output', output_name='grad_depthwise_weight', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_depthwise_w_t, int(depthwise_w_f32.size)),
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_pw1_w_t, int(pointwise1_w_f32.size)),
                (grad_pw2_w_t, int(pointwise2_w_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(depthwise_w_t.device_ptr, grad_depthwise_w_t.device_ptr, depthwise_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(depthwise_w_f32.size))
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(pointwise1_w_t.device_ptr, grad_pw1_w_t.device_ptr, pointwise1_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(pointwise1_w_f32.size))
            lib.sgd_update_fused(pointwise2_w_t.device_ptr, grad_pw2_w_t.device_ptr, pointwise2_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(pointwise2_w_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
        else:
            lib.apply_sgd_update(depthwise_w_t.device_ptr, grad_depthwise_w_t.device_ptr, float(lr), int(depthwise_w_f32.size))
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(pointwise1_w_t.device_ptr, grad_pw1_w_t.device_ptr, float(lr), int(pointwise1_w_f32.size))
            lib.apply_sgd_update(pointwise2_w_t.device_ptr, grad_pw2_w_t.device_ptr, float(lr), int(pointwise2_w_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-depthwise-layernorm2d-pointwise-gelu-pointwise-linear-training-step')
        return NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            depthwise_output=runtime.stage_to_host(depthwise_t),
            norm_output=runtime.stage_to_host(norm_t),
            pointwise1_output=runtime.stage_to_host(pw1_t),
            activation_output=runtime.stage_to_host(activation_t),
            pointwise2_output=runtime.stage_to_host(pw2_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_pointwise2_output=runtime.stage_to_host(grad_pw2_t),
            grad_activation_output=runtime.stage_to_host(grad_activation_before_gelu_t),
            grad_pointwise1_output=runtime.stage_to_host(grad_pw1_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_depthwise_output=runtime.stage_to_host(grad_depthwise_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_depthwise_weight=runtime.stage_to_host(grad_depthwise_w_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_pointwise1_weight=runtime.stage_to_host(grad_pw1_w_t),
            grad_pointwise2_weight=runtime.stage_to_host(grad_pw2_w_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_depthwise_weight=runtime.stage_to_host(depthwise_w_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_pointwise1_weight=runtime.stage_to_host(pointwise1_w_t),
            updated_pointwise2_weight=runtime.stage_to_host(pointwise2_w_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_depthwise_weight_velocity=runtime.stage_to_host(depthwise_wv_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_pointwise1_weight_velocity=runtime.stage_to_host(pointwise1_wv_t),
            updated_pointwise2_weight_velocity=runtime.stage_to_host(pointwise2_wv_t),
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
