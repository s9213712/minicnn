from __future__ import annotations

import sys
from typing import Any

from minicnn.torch_runtime import TORCH_INSTALL_HINT, import_torch_with_details, resolve_torch_device

_USER_OPERATION_ERRORS = (FileNotFoundError, TypeError, ValueError)


def format_user_error(
    problem: str,
    *,
    cause: str | None = None,
    fix: str | None = None,
    example: str | None = None,
) -> str:
    lines = [f'[ERROR] {problem}']
    if cause:
        lines.append(f'-> Cause: {cause}')
    if fix:
        lines.append(f'-> Fix: {fix}')
    if example:
        lines.append('-> Example:')
        for line in str(example).splitlines():
            lines.append(f'   {line}')
    return '\n'.join(lines)


def _exit_user_error(message: str) -> None:
    if not message.startswith('[ERROR] '):
        message = format_user_error(message)
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
        _exit_user_error(format_user_error(
            f'{command_name} requires PyTorch.',
            cause='PyTorch is not installed in this environment.',
            fix='Install the torch extra before running this command.',
            example='pip install -e .[torch]',
        ))
    if isinstance(error, ModuleNotFoundError):
        _exit_user_error(format_user_error(
            f'{command_name} could not import PyTorch because a dependency is missing.',
            cause=f'A required dependency is missing: {error.name}.',
            fix='Reinstall PyTorch for this environment or use a no-torch command.',
            example='pip install -e .[torch]',
        ))
    _exit_user_error(format_user_error(
        f'{command_name} could not import PyTorch from this environment.',
        cause=f'Import failed with: {error.__class__.__name__}: {error}',
        fix='Reinstall PyTorch or use a no-torch command.',
        example='pip install -e .[torch]',
    ))


def _ensure_torch_or_exit(command_name: str):
    return _import_torch_or_exit(command_name)


def _ensure_torch_device_supported_or_exit(cfg: dict[str, Any], command_name: str) -> None:
    torch = _import_torch_or_exit(command_name)
    train_cfg = cfg.get('train', {})
    device = str(train_cfg.get('device', 'auto'))
    try:
        resolve_torch_device(device, torch)
    except RuntimeError as exc:
        _exit_user_error(format_user_error(
            f'{command_name} cannot use train.device={device}.',
            cause=str(exc).splitlines()[0],
            fix='Use train.device=auto or train.device=cpu unless this PyTorch runtime reports CUDA availability.',
            example='minicnn train-flex --config configs/flex_cnn.yaml train.device=auto',
        ))


def _run_user_operation_or_exit(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except _USER_OPERATION_ERRORS as exc:
        _exit_user_error(format_user_error(
            'Command failed because the provided input or environment is invalid.',
            cause=str(exc),
            fix='Review the command arguments, config values, and referenced files, then retry.',
        ))
    except Exception as exc:
        _exit_internal_error(str(exc))
