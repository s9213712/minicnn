"""ctypes bindings and GPU helper functions for libminimal_cuda_cnn.so."""

import ctypes
import os
import threading

from minicnn.core._cuda_library import (
    CUDA_NATIVE_SYMBOL_GROUPS,
    REQUIRED_SYMBOLS,
    SO_PATH,
    bind_symbols,
    load_library,
    missing_symbols,
    resolve_library_path,
)
from minicnn.core._cuda_ops import (
    cnhw_to_nchw_alloc as _cnhw_to_nchw_alloc,
    conv_forward as _conv_forward,
    download_float_scalar as _download_float_scalar,
    download_int_scalar as _download_int_scalar,
    g2h as _g2h,
    gpu_scalar_float as _gpu_scalar_float,
    gpu_scalar_int as _gpu_scalar_int,
    gpu_zeros as _gpu_zeros,
    maxpool_forward as _maxpool_forward,
    nchw_to_cnhw_alloc as _nchw_to_cnhw_alloc,
    update_adam as _update_adam,
    update_conv as _update_conv,
    upload as _upload,
    upload_int as _upload_int,
    zero_bytes as _zero_bytes,
)


_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()


def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                _lib = bind_symbols(load_library())
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
        'symbol_groups': {},
        'error': None,
    }
    if not exists:
        result['error'] = 'CUDA shared library does not exist'
        return result
    try:
        candidate = bind_symbols(load_library(resolved))
        symbol_groups = {
            group_name: {
                'required_symbols': list(symbols),
                'missing_symbols': list(missing_symbols(candidate, symbols)),
            }
            for group_name, symbols in CUDA_NATIVE_SYMBOL_GROUPS.items()
        }
        for group_summary in symbol_groups.values():
            group_summary['ready'] = not bool(group_summary['missing_symbols'])
        missing = list(missing_symbols(candidate, REQUIRED_SYMBOLS))
        result['missing_symbols'] = missing
        result['symbol_groups'] = symbol_groups
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
    return _g2h(lib, ptr, size)


def gpu_zeros(size):
    return _gpu_zeros(lib, size)


def gpu_scalar_float():
    return _gpu_scalar_float(lib)


def gpu_scalar_int():
    return _gpu_scalar_int(lib)


def zero_bytes(ptr, nbytes):
    _zero_bytes(lib, ptr, nbytes)


def download_float_scalar(ptr):
    return _download_float_scalar(lib, ptr)


def download_int_scalar(ptr):
    return _download_int_scalar(lib, ptr)


def upload(arr):
    return _upload(lib, arr)


def upload_int(arr):
    return _upload_int(lib, arr)


def cnhw_to_nchw_alloc(d_cnhw, n, c, h, w):
    return _cnhw_to_nchw_alloc(lib, d_cnhw, n, c, h, w)


def nchw_to_cnhw_alloc(d_nchw, n, c, h, w):
    return _nchw_to_cnhw_alloc(lib, d_nchw, n, c, h, w)


def conv_forward(d_input_nchw, d_weight, n, in_c, in_h, in_w, out_c):
    return _conv_forward(lib, d_input_nchw, d_weight, n, in_c, in_h, in_w, out_c)


def maxpool_forward(d_input_cnhw, n, c, h, w):
    return _maxpool_forward(lib, d_input_cnhw, n, c, h, w)


def update_adam(
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
    _update_adam(
        lib,
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
        grad_normalizer=grad_normalizer,
        bias_corr1=bias_corr1,
        bias_corr2=bias_corr2,
        log_grad=log_grad,
    )


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
    _update_conv(
        lib,
        d_weight,
        d_grad,
        d_velocity,
        lr,
        momentum,
        size,
        name,
        weight_decay,
        clip_value,
        grad_normalizer=grad_normalizer,
        log_grad=log_grad,
    )
