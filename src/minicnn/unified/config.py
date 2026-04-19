from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from minicnn.config.parsing import parse_scalar
from minicnn.flex.config import _deep_update, DEFAULT_CONFIG

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
        for item in overrides:
            if '=' not in item:
                raise ValueError(f'Override must look like key=value, got: {item}')
            key, raw = item.split('=', 1)
            parts = key.split('.')
            cur = data
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = parse_scalar(raw)
    return data


def dump_unified_template() -> str:
    return yaml.safe_dump(UNIFIED_DEFAULT_CONFIG, sort_keys=False)
