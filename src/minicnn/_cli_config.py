from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from minicnn.paths import DATA_ROOT, PROJECT_ROOT
from minicnn._cli_errors import _exit_user_error


def _missing_config_message(config_path: str | None, template_cmd: str) -> str:
    shown_path = config_path or '<unspecified>'
    return (
        f'Config file not found: {shown_path}\n'
        'MiniCNN is repo-first. If built-in configs are unavailable, pass --config <path> '
        'inside a repo checkout.\n'
        f'Create a template with:\n  {template_cmd}'
    )


def _config_error_message(
    config_path: str | None,
    overrides: list[str],
    exc: Exception,
    *,
    template_cmd: str,
) -> str:
    if isinstance(exc, FileNotFoundError):
        return _missing_config_message(config_path, template_cmd)
    detail = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, yaml.YAMLError):
        shown_path = config_path or '<unspecified>'
        return f'Failed to parse config file: {shown_path}\n{detail}'
    if 'mapping at the top level' in detail:
        shown_path = config_path or '<unspecified>'
        return f'Failed to parse config file: {shown_path}\n{detail}'
    if (
        'Override must look like key=value' in detail
        or detail.startswith('Override path ')
        or detail.startswith('Override key cannot be empty')
    ):
        if overrides:
            return f'Invalid config override: {overrides[-1]}\n{detail}'
        return f'Invalid config override.\n{detail}'
    shown_path = config_path or '<unspecified>'
    return f'Invalid config file or override for: {shown_path}\n{detail}'


def _resolve_cli_config_path(config_path: str | None) -> str | None:
    if not config_path:
        return config_path
    raw_path = Path(config_path)
    if raw_path.exists():
        return str(raw_path)
    if not raw_path.is_absolute():
        project_relative = PROJECT_ROOT / raw_path
        if project_relative.exists():
            return str(project_relative)
    return config_path


def _load_config_or_exit(loader, config_path: str | None, overrides: list[str], *, template_cmd: str) -> dict:
    resolved_path = _resolve_cli_config_path(config_path)
    try:
        return loader(resolved_path, overrides)
    except (FileNotFoundError, TypeError, ValueError, IndexError, yaml.YAMLError) as exc:
        _exit_user_error(
            _config_error_message(config_path, overrides, exc, template_cmd=template_cmd)
        )


def _load_flex_config_or_exit(config_path: str | None, overrides: list[str]) -> dict:
    from minicnn.flex.config import load_flex_config

    return _load_config_or_exit(
        load_flex_config,
        config_path,
        overrides,
        template_cmd='minicnn config-template > configs/my_config.yaml',
    )


def _load_unified_config_or_exit(config_path: str | None, overrides: list[str]) -> dict:
    from minicnn.unified.config import load_unified_config

    return _load_config_or_exit(
        load_unified_config,
        config_path,
        overrides,
        template_cmd='minicnn dual-config-template > configs/my_config.yaml',
    )


def _ensure_cuda_legacy_prereqs_or_exit(cfg: dict[str, Any]) -> None:
    from minicnn.core.cuda_backend import resolve_library_path
    from minicnn.data.cifar10 import cifar10_ready

    library_path = Path(resolve_library_path())
    if not library_path.exists():
        _exit_user_error(
            'cuda_legacy training requires a native CUDA shared library.\n'
            'Build it with:\n'
            '  minicnn build --legacy-make --check'
        )
    dataset_cfg = cfg.get('dataset', {})
    if str(dataset_cfg.get('type', 'cifar10')) == 'cifar10':
        data_root = Path(dataset_cfg.get('data_root', DATA_ROOT))
        if not cifar10_ready(data_root):
            _exit_user_error(
                'cuda_legacy training requires prepared CIFAR-10 data.\n'
                'Prepare it with:\n'
                '  minicnn prepare-data'
            )
