"""High-level GPU buffer helpers built on top of the loaded CUDA library."""

from __future__ import annotations

from ctypes import c_float

import numpy as np

from minicnn.config.settings import KH, KW, LEAKY_ALPHA


def g2h(bound_lib, ptr, size):
    h = np.zeros(size, dtype=np.float32)
    bound_lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h


def gpu_zeros(bound_lib, size):
    ptr = bound_lib.gpu_malloc(size * 4)
    bound_lib.gpu_memset(ptr, 0, size * 4)
    return ptr


def gpu_scalar_float(bound_lib):
    return bound_lib.gpu_malloc(4)


def gpu_scalar_int(bound_lib):
    return bound_lib.gpu_malloc(4)


def zero_bytes(bound_lib, ptr, nbytes):
    bound_lib.gpu_memset(ptr, 0, nbytes)


def download_float_scalar(bound_lib, ptr):
    h = np.zeros(1, dtype=np.float32)
    bound_lib.gpu_memcpy_d2h(h.ctypes.data, ptr, 4)
    return float(h[0])


def download_int_scalar(bound_lib, ptr):
    h = np.zeros(1, dtype=np.int32)
    bound_lib.gpu_memcpy_d2h(h.ctypes.data, ptr, 4)
    return int(h[0])


def upload(bound_lib, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    ptr = bound_lib.gpu_malloc(arr.size * 4)
    bound_lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)
    return ptr


def upload_int(bound_lib, arr):
    arr = np.ascontiguousarray(arr.astype(np.int32, copy=False))
    ptr = bound_lib.gpu_malloc(arr.size * 4)
    bound_lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)
    return ptr


def cnhw_to_nchw_alloc(bound_lib, d_cnhw, n, c, h, w):
    d_nchw = bound_lib.gpu_malloc(n * c * h * w * 4)
    bound_lib.cnhw_to_nchw(d_cnhw, d_nchw, n, c, h, w)
    return d_nchw


def nchw_to_cnhw_alloc(bound_lib, d_nchw, n, c, h, w):
    d_cnhw = bound_lib.gpu_malloc(n * c * h * w * 4)
    bound_lib.nchw_to_cnhw(d_nchw, d_cnhw, n, c, h, w)
    return d_cnhw


def conv_forward(bound_lib, d_input_nchw, d_weight, n, in_c, in_h, in_w, out_c):
    out_h, out_w = in_h - KH + 1, in_w - KW + 1
    col_size = in_c * KH * KW * n * out_h * out_w
    raw_size = out_c * n * out_h * out_w
    d_col = bound_lib.gpu_malloc(col_size * 4)
    d_raw = bound_lib.gpu_malloc(raw_size * 4)
    bound_lib.im2col_forward(d_input_nchw, d_col, n, in_c, in_h, in_w, KH, KW, out_h, out_w)
    bound_lib.gemm_forward(d_weight, d_col, d_raw, out_c, n * out_h * out_w, in_c * KH * KW)
    bound_lib.leaky_relu_forward(d_raw, c_float(LEAKY_ALPHA), raw_size)
    return d_col, d_raw, out_h, out_w


def maxpool_forward(bound_lib, d_input_cnhw, n, c, h, w):
    out_h, out_w = h // 2, w // 2
    out_size = c * n * out_h * out_w
    d_pool = bound_lib.gpu_malloc(out_size * 4)
    d_idx = bound_lib.gpu_malloc(out_size * 4)
    bound_lib.maxpool_forward_store(d_pool, d_input_cnhw, d_idx, n, c, h, w)
    return d_pool, d_idx, out_h, out_w


def update_adam(
    bound_lib,
    d_weight,
    d_grad,
    d_m,
    d_v,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    clip_value,
    size,
    name,
    grad_normalizer=1.0,
    bias_corr1=1.0,
    bias_corr2=1.0,
    log_grad=False,
):
    if log_grad:
        h_grad = g2h(bound_lib, d_grad, size).reshape(-1) / grad_normalizer
        h_weight = g2h(bound_lib, d_weight, size).reshape(-1)
        h_grad = h_grad + weight_decay * h_weight
        print(f"    {name} grad_abs_mean={np.mean(np.abs(h_grad)):.6e} grad_abs_max={np.max(np.abs(h_grad)):.6e}")

    bound_lib.adam_update_fused(
        d_weight, d_grad, d_m, d_v,
        c_float(lr), c_float(beta1), c_float(beta2), c_float(eps),
        c_float(weight_decay), c_float(clip_value),
        c_float(grad_normalizer), c_float(bias_corr1), c_float(bias_corr2),
        size,
    )


def update_conv(
    bound_lib,
    d_weight,
    d_grad,
    d_velocity,
    lr,
    momentum,
    size,
    name,
    weight_decay,
    clip_value,
    grad_normalizer=1.0,
    log_grad=False,
):
    if log_grad:
        h_grad = g2h(bound_lib, d_grad, size).reshape(-1)
        h_grad = h_grad / grad_normalizer
        h_weight = g2h(bound_lib, d_weight, size).reshape(-1)
        h_grad = h_grad + weight_decay * h_weight
        print(f"    {name} grad_abs_mean={np.mean(np.abs(h_grad)):.6e} grad_abs_max={np.max(np.abs(h_grad)):.6e}")

    bound_lib.conv_update_fused(
        d_weight,
        d_grad,
        d_velocity,
        c_float(lr),
        c_float(momentum),
        c_float(weight_decay),
        c_float(clip_value),
        c_float(grad_normalizer),
        size,
    )
