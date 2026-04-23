from __future__ import annotations

import sys
from typing import Any

from minicnn.torch_runtime import TORCH_INSTALL_HINT, import_torch_with_details, resolve_torch_device

_USER_OPERATION_ERRORS = (FileNotFoundError, TypeError, ValueError)


def _exit_user_error(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _exit_internal_error(message: str) -> None:
    print(f'Internal error: {message}', file=sys.stderr)
    raise SystemExit(1)


def _import_torch_or_exit(command_name: str):
    torch, error = import_torch_with_details()
    if torch is not None:
        return torch
    if error is None:
        _exit_user_error(f'{command_name} requires PyTorch.\n{TORCH_INSTALL_HINT}')
    if isinstance(error, ModuleNotFoundError):
        _exit_user_error(
            f'{command_name} could not import PyTorch because a dependency is missing: {error.name}.\n'
            f'Reinstall PyTorch for this environment.\n{TORCH_INSTALL_HINT}'
        )
    _exit_user_error(
        f'{command_name} could not import PyTorch from this environment.\n'
        f'Import failed with: {error.__class__.__name__}: {error}\n'
        f'Reinstall PyTorch or use a no-torch command.\n{TORCH_INSTALL_HINT}'
    )


def _ensure_torch_or_exit(command_name: str):
    return _import_torch_or_exit(command_name)


def _ensure_torch_device_supported_or_exit(cfg: dict[str, Any], command_name: str) -> None:
    torch = _import_torch_or_exit(command_name)
    train_cfg = cfg.get('train', {})
    device = str(train_cfg.get('device', 'auto'))
    try:
        resolve_torch_device(device, torch)
    except RuntimeError as exc:
        _exit_user_error(
            str(exc)
        )


def _run_user_operation_or_exit(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except _USER_OPERATION_ERRORS as exc:
        _exit_user_error(str(exc))
    except Exception as exc:
        _exit_internal_error(str(exc))
