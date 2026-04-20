"""ctypes bindings and GPU helper functions for libminimal_cuda_cnn.so."""

import ctypes
import os
from pathlib import Path
from ctypes import c_float, c_int, c_void_p

import numpy as np

from minicnn.config.settings import KH, KW, LEAKY_ALPHA


from minicnn.paths import CPP_ROOT, PROJECT_ROOT

if os.name == 'nt':
    NATIVE_VARIANTS = {
        'default': 'minimal_cuda_cnn.dll',
        'cublas': 'minimal_cuda_cnn_cublas.dll',
        'handmade': 'minimal_cuda_cnn_handmade.dll',
        'nocublas': 'minimal_cuda_cnn_handmade.dll',
    }
else:
    NATIVE_VARIANTS = {
        'default': 'libminimal_cuda_cnn.so',
        'cublas': 'libminimal_cuda_cnn_cublas.so',
        'handmade': 'libminimal_cuda_cnn_handmade.so',
        'nocublas': 'libminimal_cuda_cnn_handmade.so',
    }
SO_PATH = str(CPP_ROOT / NATIVE_VARIANTS['default'])
REQUIRED_SYMBOLS = (
    'gpu_malloc',
    'gpu_free',
    'gpu_memcpy_h2d',
    'gpu_memcpy_d2h',
    'im2col_forward',
    'gemm_forward',
    'dense_forward',
    'softmax_xent_grad_loss_acc',
)


def resolve_library_path(path: str | os.PathLike[str] | None = None) -> str:
    if path is None:
        path = os.environ.get('MINICNN_CUDA_SO') or os.environ.get('MINICNN_SO_PATH')
    if path is not None:
        raw = Path(path)
        if raw.name == str(path) and raw.name in NATIVE_VARIANTS:
            return str(CPP_ROOT / NATIVE_VARIANTS[raw.name])
        if raw.is_absolute():
            return str(raw)
        if raw.parts and raw.parts[0] == 'cpp':
            return str(PROJECT_ROOT / raw)
        return str(CPP_ROOT / raw)

    variant = os.environ.get('MINICNN_CUDA_VARIANT', 'default').lower()
    if variant not in NATIVE_VARIANTS:
        choices = ', '.join(sorted(NATIVE_VARIANTS))
        raise RuntimeError(f"Unknown MINICNN_CUDA_VARIANT={variant!r}; expected one of: {choices}")
    return str(CPP_ROOT / NATIVE_VARIANTS[variant])


def load_library(path: str | os.PathLike[str] | None = None):
    path = resolve_library_path(path)
    if not os.path.exists(path):
        raise RuntimeError(
            "CUDA shared library not found:\n"
            f"  {path}\n\n"
            "Build it from the repository root first:\n"
            "  make -C cpp\n"
            "  make -C cpp variants\n\n"
            "Optional backends:\n"
            "  make -C cpp cublas    # cuBLAS GEMM path\n"
            "  make -C cpp handmade  # handwritten CUDA fallback\n\n"
            "Select at runtime with:\n"
            "  MINICNN_CUDA_VARIANT=cublas\n"
            "  MINICNN_CUDA_VARIANT=handmade\n"
            "  MINICNN_CUDA_SO=cpp/libminimal_cuda_cnn_cublas.so\n"
        )
    try:
        if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(str(CPP_ROOT))
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                os.add_dll_directory(str(Path(cuda_path) / 'bin'))
        return ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load CUDA shared library:\n"
            f"  {path}\n\n"
            f"Loader error: {exc}\n\n"
            "Check that CUDA runtime libraries are visible, then rebuild:\n"
            "  make -C cpp\n"
        ) from exc


_lib: ctypes.CDLL | None = None


def _bind_symbols(bound_lib: ctypes.CDLL) -> ctypes.CDLL:
    bound_lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    bound_lib.gpu_malloc.restype = c_void_p
    bound_lib.gpu_free.argtypes = [c_void_p]
    bound_lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
    bound_lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
    bound_lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
    if hasattr(bound_lib, 'gpu_synchronize'):
        bound_lib.gpu_synchronize.argtypes = []
        bound_lib.gpu_synchronize.restype = None
    bound_lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
    bound_lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
    bound_lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
    bound_lib.leaky_relu_backward.argtypes = [c_void_p, c_void_p, c_float, c_int]
    bound_lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
    bound_lib.dense_backward_full.argtypes = [
        c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int,
    ]
    bound_lib.softmax_xent_grad_loss_acc.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int,
    ]
    bound_lib.count_correct.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]
    bound_lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
    bound_lib.apply_momentum_update.argtypes = [c_void_p, c_void_p, c_void_p, c_float, c_float, c_int]
    bound_lib.conv_update_fused.argtypes = [
        c_void_p, c_void_p, c_void_p,
        c_float, c_float, c_float, c_float, c_float,
        c_int,
    ]
    bound_lib.clip_inplace.argtypes = [c_void_p, c_float, c_int]
    bound_lib.nchw_to_cnhw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    bound_lib.cnhw_to_nchw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    bound_lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    bound_lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    if hasattr(bound_lib, 'maxpool_backward_nchw'):
        bound_lib.maxpool_backward_nchw.argtypes = [
            c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int,
        ]
    if hasattr(bound_lib, 'maxpool_backward_nchw_status'):
        bound_lib.maxpool_backward_nchw_status.argtypes = [
            c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int,
        ]
        bound_lib.maxpool_backward_nchw_status.restype = c_int
    bound_lib.conv_backward.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
    ]
    bound_lib.conv_backward_precol.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
    ]
    return bound_lib


def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _bind_symbols(load_library())
    return _lib


def reset_library_cache() -> None:
    """Force the next CUDA call to resolve and load the current configured library."""
    global _lib
    _lib = None


def is_lib_loaded() -> bool:
    """Return True if the native CUDA library has already been loaded into this process."""
    return _lib is not None


def check_cuda_ready(path: str | os.PathLike[str] | None = None) -> dict[str, object]:
    resolved = resolve_library_path(path)
    exists = os.path.exists(resolved)
    result: dict[str, object] = {
        'path': resolved,
        'exists': exists,
        'loadable': False,
        'missing_symbols': [],
        'error': None,
    }
    if not exists:
        result['error'] = 'CUDA shared library does not exist'
        return result
    try:
        candidate = _bind_symbols(load_library(resolved))
        missing = [name for name in REQUIRED_SYMBOLS if not hasattr(candidate, name)]
        result['missing_symbols'] = missing
        result['loadable'] = not missing
        if missing:
            result['error'] = f'Missing required symbols: {missing}'
    except Exception as exc:
        result['error'] = str(exc)
    return result


class LazyCudaLibrary:
    def __getattr__(self, name: str):
        return getattr(get_lib(), name)


lib = LazyCudaLibrary()


def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h


def gpu_zeros(size):
    ptr = lib.gpu_malloc(size * 4)
    lib.gpu_memset(ptr, 0, size * 4)
    return ptr


def gpu_scalar_float():
    return lib.gpu_malloc(4)


def gpu_scalar_int():
    return lib.gpu_malloc(4)


def zero_bytes(ptr, nbytes):
    lib.gpu_memset(ptr, 0, nbytes)


def download_float_scalar(ptr):
    h = np.zeros(1, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, 4)
    return float(h[0])


def download_int_scalar(ptr):
    h = np.zeros(1, dtype=np.int32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, 4)
    return int(h[0])


def upload(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    ptr = lib.gpu_malloc(arr.size * 4)
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)
    return ptr


def upload_int(arr):
    arr = np.ascontiguousarray(arr.astype(np.int32, copy=False))
    ptr = lib.gpu_malloc(arr.size * 4)
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)
    return ptr


def cnhw_to_nchw_alloc(d_cnhw, n, c, h, w):
    d_nchw = lib.gpu_malloc(n * c * h * w * 4)
    lib.cnhw_to_nchw(d_cnhw, d_nchw, n, c, h, w)
    return d_nchw


def nchw_to_cnhw_alloc(d_nchw, n, c, h, w):
    d_cnhw = lib.gpu_malloc(n * c * h * w * 4)
    lib.nchw_to_cnhw(d_nchw, d_cnhw, n, c, h, w)
    return d_cnhw


def conv_forward(d_input_nchw, d_weight, n, in_c, in_h, in_w, out_c):
    out_h, out_w = in_h - KH + 1, in_w - KW + 1
    col_size = in_c * KH * KW * n * out_h * out_w
    raw_size = out_c * n * out_h * out_w
    d_col = lib.gpu_malloc(col_size * 4)
    d_raw = lib.gpu_malloc(raw_size * 4)
    lib.im2col_forward(d_input_nchw, d_col, n, in_c, in_h, in_w, KH, KW, out_h, out_w)
    lib.gemm_forward(d_weight, d_col, d_raw, out_c, n * out_h * out_w, in_c * KH * KW)
    lib.leaky_relu_forward(d_raw, c_float(LEAKY_ALPHA), raw_size)
    return d_col, d_raw, out_h, out_w


def maxpool_forward(d_input_cnhw, n, c, h, w):
    out_h, out_w = h // 2, w // 2
    out_size = c * n * out_h * out_w
    d_pool = lib.gpu_malloc(out_size * 4)
    d_idx = lib.gpu_malloc(out_size * 4)
    lib.maxpool_forward_store(d_pool, d_input_cnhw, d_idx, n, c, h, w)
    return d_pool, d_idx, out_h, out_w


def update_conv(
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
        h_grad = g2h(d_grad, size).reshape(-1)
        h_grad = h_grad / grad_normalizer
        h_weight = g2h(d_weight, size).reshape(-1)
        h_grad = h_grad + weight_decay * h_weight
        print(f"    {name} grad_abs_mean={np.mean(np.abs(h_grad)):.6e} grad_abs_max={np.max(np.abs(h_grad)):.6e}")

    lib.conv_update_fused(
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
