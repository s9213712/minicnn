from __future__ import annotations

import importlib
from typing import Any


_TORCH_INSTALL_HINT = 'Install it with:\n  pip install -e .[torch]'


def _exit_user_error(message: str) -> None:
    print(message)
    raise SystemExit(2)


def _import_torch_or_exit(command_name: str):
    try:
        return importlib.import_module('torch')
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            _exit_user_error(f'{command_name} requires PyTorch.\n{_TORCH_INSTALL_HINT}')
        _exit_user_error(
            f'{command_name} could not import PyTorch because a dependency is missing: {exc.name}.\n'
            f'Reinstall PyTorch for this environment.\n{_TORCH_INSTALL_HINT}'
        )
    except Exception as exc:
        _exit_user_error(
            f'{command_name} could not import PyTorch from this environment.\n'
            f'Import failed with: {exc.__class__.__name__}: {exc}\n'
            f'Reinstall PyTorch or use a no-torch command.\n{_TORCH_INSTALL_HINT}'
        )


def _ensure_torch_or_exit(command_name: str):
    return _import_torch_or_exit(command_name)


def _ensure_torch_device_supported_or_exit(cfg: dict[str, Any], command_name: str) -> None:
    torch = _import_torch_or_exit(command_name)
    train_cfg = cfg.get('train', {})
    device = str(train_cfg.get('device', 'auto'))
    if device == 'cuda' and not torch.cuda.is_available():
        _exit_user_error(
            'Requested train.device=cuda, but CUDA is not available in this PyTorch runtime.\n'
            'Use train.device=auto or train.device=cpu.'
        )


def _run_user_operation_or_exit(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        _exit_user_error(str(exc))
