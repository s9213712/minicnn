"""Small CUDA helper operations used by the legacy CIFAR-10 training loop."""

from ctypes import c_float

import numpy as np

from minicnn.config.settings import KH, KW, LEAKY_ALPHA
from minicnn.core.cuda_backend import lib


def malloc_floats(size):
    return lib.gpu_malloc(size * 4)


def upload_to(ptr, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)


def upload_int_to(ptr, arr):
    arr = np.ascontiguousarray(arr.astype(np.int32, copy=False))
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)


def zero_floats(ptr, size):
    lib.gpu_memset(ptr, 0, size * 4)


def conv_forward_into(d_input_nchw, d_weight, d_col, d_raw, n, in_c, in_h, in_w, out_c):
    out_h, out_w = in_h - KH + 1, in_w - KW + 1
    lib.im2col_forward(d_input_nchw, d_col, n, in_c, in_h, in_w, KH, KW, out_h, out_w)
    lib.gemm_forward(d_weight, d_col, d_raw, out_c, n * out_h * out_w, in_c * KH * KW)
    lib.leaky_relu_forward(d_raw, c_float(LEAKY_ALPHA), out_c * n * out_h * out_w)


def maxpool_forward_into(d_input_cnhw, d_pool, d_max_idx, n, c, h, w):
    lib.maxpool_forward_store(d_pool, d_input_cnhw, d_max_idx, n, c, h, w)


def cnhw_to_nchw_into(d_cnhw, d_nchw, n, c, h, w):
    lib.cnhw_to_nchw(d_cnhw, d_nchw, n, c, h, w)


def nchw_to_cnhw_into(d_nchw, d_cnhw, n, c, h, w):
    lib.nchw_to_cnhw(d_nchw, d_cnhw, n, c, h, w)
