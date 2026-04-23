from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from minicnn.config.parsing import parse_override_parts, parse_scalar, set_nested_value
from minicnn.flex.config import _deep_update, _normalize_repo_relative_paths, DEFAULT_CONFIG

UNIFIED_DEFAULT_CONFIG: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
UNIFIED_DEFAULT_CONFIG['engine'] = {
    'backend': 'torch',  # torch | cuda_legacy
    'strict_backend_validation': True,
}
UNIFIED_DEFAULT_CONFIG.setdefault('runtime', {}).update({'save_config': True})


def load_unified_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    data = copy.deepcopy(UNIFIED_DEFAULT_CONFIG)
    if path:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError('Config file must contain a mapping at the top level')
        _deep_update(data, loaded)
    if overrides:
        parsed_overrides: list[tuple[list[str], Any]] = []
        for item in overrides:
            parts, raw = parse_override_parts(item)
            parsed_overrides.append((parts, parse_scalar(raw)))

        parsed_overrides.sort(key=lambda item: 0 if item[0][-1] == 'type' else 1)
        for parts, value in parsed_overrides:
            set_nested_value(data, parts, value, clear_on_type_change=True)
    return _normalize_repo_relative_paths(data)


def dump_unified_template() -> str:
    return yaml.safe_dump(UNIFIED_DEFAULT_CONFIG, sort_keys=False)
