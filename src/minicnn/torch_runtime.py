from __future__ import annotations

import importlib
from typing import Any


TORCH_INSTALL_HINT = 'Install it with:\n  pip install -e .[torch]'


def import_torch_with_details() -> tuple[Any | None, Exception | None]:
    try:
        return importlib.import_module('torch'), None
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            return None, None
        return None, exc
    except Exception as exc:  # pragma: no cover
        return None, exc


def format_torch_import_error(command_name: str, *, action: str) -> str:
    _torch, error = import_torch_with_details()
    if error is None and _torch is None:
        return f'{command_name} requires PyTorch to {action}.\n{TORCH_INSTALL_HINT}'
    if isinstance(error, ModuleNotFoundError):
        return (
            f'{command_name} could not import PyTorch because a dependency is missing: {error.name}.\n'
            f'Reinstall PyTorch for this environment.\n{TORCH_INSTALL_HINT}'
        )
    return (
        f'{command_name} could not import PyTorch from this environment.\n'
        f'Import failed with: {error.__class__.__name__}: {error}\n'
        f'Reinstall PyTorch or use a no-torch command.\n{TORCH_INSTALL_HINT}'
    )


def require_torch(command_name: str, *, action: str):
    torch, error = import_torch_with_details()
    if torch is not None:
        return torch
    raise RuntimeError(format_torch_import_error(command_name, action=action))


def resolve_torch_device(requested: str, torch_module) -> Any:
    if requested == 'cpu':
        return torch_module.device('cpu')
    if requested == 'cuda':
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                'Requested train.device=cuda, but CUDA is not available in this PyTorch runtime.\n'
                'Use train.device=auto or train.device=cpu.'
            )
        return torch_module.device('cuda')
    return torch_module.device('cuda' if torch_module.cuda.is_available() else 'cpu')
