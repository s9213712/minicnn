"""CUDA shared-library resolution and symbol binding helpers."""

from __future__ import annotations

import ctypes
import os
from ctypes import c_float, c_int, c_void_p
from pathlib import Path

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
    'apply_relu',
    'apply_maxpool',
    'add_forward',
    'concat_forward',
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
    resolved = resolve_library_path(path)
    if not os.path.exists(resolved):
        raise RuntimeError(
            "CUDA shared library not found:\n"
            f"  {resolved}\n\n"
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
        return ctypes.CDLL(resolved)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load CUDA shared library:\n"
            f"  {resolved}\n\n"
            f"Loader error: {exc}\n\n"
            "Check that CUDA runtime libraries are visible, then rebuild:\n"
            "  make -C cpp\n"
        ) from exc


def bind_symbols(bound_lib: ctypes.CDLL) -> ctypes.CDLL:
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
    bound_lib.apply_relu.argtypes = [c_void_p, c_int]
    if hasattr(bound_lib, 'apply_relu_backward'):
        bound_lib.apply_relu_backward.argtypes = [c_void_p, c_void_p, c_int]
    bound_lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    if hasattr(bound_lib, 'add_forward'):
        bound_lib.add_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
    if hasattr(bound_lib, 'concat_forward'):
        bound_lib.concat_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
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
    if hasattr(bound_lib, 'mse_fwd_grad_loss_acc'):
        bound_lib.mse_fwd_grad_loss_acc.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int,
        ]
    if hasattr(bound_lib, 'bce_fwd_grad_loss_acc'):
        bound_lib.bce_fwd_grad_loss_acc.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int,
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
    if hasattr(bound_lib, 'layer_norm_forward'):
        bound_lib.layer_norm_forward.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_float,
        ]
    if hasattr(bound_lib, 'layer_norm_backward'):
        bound_lib.layer_norm_backward.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_float,
        ]
    if hasattr(bound_lib, 'bn_train_forward'):
        bound_lib.bn_train_forward.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int, c_int,
            c_float, c_float,
        ]
    if hasattr(bound_lib, 'bn_eval_forward'):
        bound_lib.bn_eval_forward.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int, c_int, c_float,
        ]
    if hasattr(bound_lib, 'bn_backward'):
        bound_lib.bn_backward.argtypes = [
            c_void_p, c_void_p, c_void_p,
            c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int, c_int,
        ]
    if hasattr(bound_lib, 'adam_update_fused'):
        bound_lib.adam_update_fused.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p,
            c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float,
            c_int,
        ]
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
