from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from minicnn.config.parsing import parse_override_parts, parse_scalar, set_nested_value
from minicnn.paths import PROJECT_ROOT


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
        'seed': 42,
        'init_seed': 42,
        'early_stop_patience': 0,
        'min_delta': 0.0,
    },
    'augmentation': {
        'normalize': True,
        'random_crop': False,
        'random_crop_padding': 4,
        'horizontal_flip': False,
    },
    'loss': {'type': 'CrossEntropyLoss'},
    'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'exclude_bias_norm_weight_decay': True},
    'scheduler': {'enabled': False, 'type': 'StepLR', 'step_size': 10, 'gamma': 0.5},
    'runtime': {'save_every_n_epochs': 0},
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


def _normalize_repo_relative_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get('dataset')
    if isinstance(dataset_cfg, dict):
        data_root = dataset_cfg.get('data_root')
        if isinstance(data_root, str) and data_root.strip():
            candidate = Path(data_root).expanduser()
            if not candidate.is_absolute():
                dataset_cfg['data_root'] = str((PROJECT_ROOT / candidate).resolve())
    return cfg


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
            parts, raw = parse_override_parts(item)
            parsed_overrides.append((parts, parse_scalar(raw)))

        parsed_overrides.sort(key=lambda item: 0 if item[0][-1] == 'type' else 1)
        for parts, value in parsed_overrides:
            set_nested_value(data, parts, value, clear_on_type_change=True)
    return _normalize_repo_relative_paths(data)


def dump_template() -> str:
    return yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False)
