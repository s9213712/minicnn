from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    'project': {
        'name': 'minicnn-flex',
        'run_name': 'default',
        'artifacts_root': 'artifacts',
    },
    'dataset': {
        'type': 'cifar10',
        'data_root': 'data/cifar-10-batches-py',
        'download': False,
        'num_samples': 512,
        'val_samples': 128,
        'num_classes': 10,
        'input_shape': [3, 32, 32],
        'seed': 42,
    },
    'model': {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ],
    },
    'train': {
        'epochs': 1,
        'batch_size': 64,
        'device': 'auto',
        'amp': False,
        'grad_accum_steps': 1,
        'num_workers': 0,
        'log_every': 10,
    },
    'loss': {'type': 'CrossEntropyLoss'},
    'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9},
    'scheduler': {'enabled': False, 'type': 'StepLR', 'step_size': 10, 'gamma': 0.5},
}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if (
            isinstance(v, dict)
            and isinstance(dst.get(k), dict)
            and 'type' in v
            and 'type' in dst[k]
            and v['type'] != dst[k]['type']
        ):
            dst[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _parse_scalar(text: str) -> Any:
    low = text.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    if low in {'none', 'null'}:
        return None
    try:
        if text.startswith('0') and text not in {'0', '0.0'} and not text.startswith('0.'):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith('[') and text.endswith(']'):
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, list):
            return loaded
    return text


def load_flex_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    data = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError('Config file must contain a mapping at the top level')
        _deep_update(data, loaded)
    if overrides:
        parsed_overrides: list[tuple[list[str], Any]] = []
        for item in overrides:
            if '=' not in item:
                raise ValueError(f'Override must look like key=value, got: {item}')
            key, raw = item.split('=', 1)
            parsed_overrides.append((key.split('.'), _parse_scalar(raw)))

        parsed_overrides.sort(key=lambda item: 0 if item[0][-1] == 'type' else 1)
        for parts, value in parsed_overrides:
            cur = data
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            if parts[-1] == 'type' and isinstance(cur, dict) and cur.get('type') != value:
                cur.clear()
            cur[parts[-1]] = value
    return data


def dump_template() -> str:
    return yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False)
