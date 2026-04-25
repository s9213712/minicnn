"""ctypes bindings and GPU helper functions for libminimal_cuda_cnn.so."""

import ctypes
import os
import threading
from collections.abc import Callable

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


def _format_cuda_runtime_version(raw_version: object) -> str:
    try:
        value = int(raw_version)
    except (TypeError, ValueError):
        return 'unknown'
    if value <= 0:
        return 'unknown'
    return f'{value // 1000}.{(value % 1000) // 10}'


def _is_wsl_environment() -> bool:
    if os.environ.get('WSL_INTEROP') or os.environ.get('WSL_DISTRO_NAME'):
        return True
    try:
        with open('/proc/version', encoding='utf-8') as handle:
            return 'microsoft' in handle.read().lower()
    except OSError:
        return False


def _cuda_environment_diagnostics(
    runtime_preflight: dict[str, object],
    *,
    exists_fn: Callable[[str], bool] = os.path.exists,
    realpath_fn: Callable[[str], str] = os.path.realpath,
) -> dict[str, object]:
    status = runtime_preflight.get('status')
    driver_version = int(runtime_preflight.get('driver_version') or 0)
    runtime_version = int(runtime_preflight.get('runtime_version') or 0)
    runtime_driver_mismatch = (
        status == 35
        or (driver_version > 0 and runtime_version > 0 and driver_version < runtime_version)
    )
    wsl = _is_wsl_environment()
    device_nodes = {
        '/dev/dxg': exists_fn('/dev/dxg'),
        '/dev/nvidiactl': exists_fn('/dev/nvidiactl'),
        '/dev/nvidia0': exists_fn('/dev/nvidia0'),
    }
    libcuda_candidates = []
    for path in (
        '/usr/lib/wsl/lib/libcuda.so.1',
        '/lib/x86_64-linux-gnu/libcuda.so.1',
        '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
    ):
        exists = exists_fn(path)
        libcuda_candidates.append({
            'path': path,
            'exists': exists,
            'resolved_path': realpath_fn(path) if exists else None,
        })

    remediation: list[str] = []
    issue = None
    if wsl and not device_nodes['/dev/dxg']:
        issue = 'wsl_cuda_device_node_missing'
        remediation.append(
            'Restart WSL and ensure the Windows NVIDIA driver exposes /dev/dxg inside the distro.'
        )
    if runtime_driver_mismatch:
        issue = issue or 'cuda_driver_runtime_mismatch'
        remediation.append(
            'Use a CUDA runtime/toolkit compatible with the driver visible to the process '
            f'(driver={_format_cuda_runtime_version(driver_version)}, '
            f'runtime={_format_cuda_runtime_version(runtime_version)}).'
        )
    if driver_version <= 0:
        issue = issue or 'cuda_driver_api_unavailable'
        remediation.append(
            'Ensure libcuda.so.1 resolves to the active NVIDIA driver rather than a stale stub.'
        )
    if wsl:
        remediation.append(
            'In WSL, prefer the Windows-provided CUDA driver path /usr/lib/wsl/lib and avoid stale Linux NVIDIA driver packages shadowing it.'
        )

    return {
        'wsl': wsl,
        'ld_library_path': os.environ.get('LD_LIBRARY_PATH', ''),
        'cuda_home': os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'),
        'device_nodes': device_nodes,
        'libcuda_candidates': libcuda_candidates,
        'runtime_driver_mismatch': runtime_driver_mismatch,
        'issue': issue,
        'remediation': remediation,
    }


def check_cuda_ready(path: str | os.PathLike[str] | None = None) -> dict[str, object]:
    resolved = resolve_library_path(path)
    exists = os.path.exists(resolved)
    result: dict[str, object] = {
        'path': resolved,
        'exists': exists,
        'loadable': False,
        'ready': False,
        'missing_symbols': [],
        'symbol_groups': {},
        'runtime_preflight': {
            'available': False,
            'ready': None,
            'status': None,
            'status_message': None,
            'driver_version': None,
            'runtime_version': None,
        },
        'environment_diagnostics': {},
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
        runtime_ready = True
        if hasattr(candidate, 'cuda_runtime_status'):
            status = int(candidate.cuda_runtime_status())
            status_message = 'unknown CUDA runtime error'
            if hasattr(candidate, 'cuda_runtime_status_string'):
                raw_status_message = candidate.cuda_runtime_status_string(status)
                if raw_status_message:
                    status_message = raw_status_message.decode('utf-8', errors='replace')
            driver_version = (
                int(candidate.cuda_runtime_driver_version())
                if hasattr(candidate, 'cuda_runtime_driver_version')
                else 0
            )
            runtime_version = (
                int(candidate.cuda_runtime_version())
                if hasattr(candidate, 'cuda_runtime_version')
                else 0
            )
            runtime_ready = status == 0
            result['runtime_preflight'] = {
                'available': True,
                'ready': runtime_ready,
                'status': status,
                'status_message': status_message,
                'driver_version': driver_version,
                'runtime_version': runtime_version,
            }
            result['environment_diagnostics'] = _cuda_environment_diagnostics(
                result['runtime_preflight']
            )
        if missing:
            result['error'] = f'Missing required symbols: {missing}'
        elif not runtime_ready:
            runtime_summary = result['runtime_preflight']
            diagnostics = result['environment_diagnostics']
            remediation = diagnostics.get('remediation') if isinstance(diagnostics, dict) else None
            remediation_text = ''
            if remediation:
                remediation_text = ' Remediation: ' + ' '.join(str(item) for item in remediation)
            result['error'] = (
                'CUDA runtime preflight failed: '
                f"{runtime_summary['status_message']} "
                f"(status={runtime_summary['status']}, "
                f"driver={runtime_summary['driver_version']}, "
                f"runtime={runtime_summary['runtime_version']})."
                f"{remediation_text}"
            )
        result['ready'] = bool(result['loadable']) and runtime_ready
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
