from __future__ import annotations

from pathlib import Path

import yaml

from minicnn.flex.builder import build_model
from minicnn.flex.config import load_flex_config


TEMPLATES_ROOT = Path(__file__).resolve().parents[1] / 'templates'
TRAIN_POOL_LIMITS = {
    'mnist': 60000,
    'cifar10': 50000,
}


def _validate_dataset_split(cfg: dict, path: Path) -> None:
    dataset_cfg = cfg.get('dataset', {})
    dataset_type = str(dataset_cfg.get('type', ''))
    limit = TRAIN_POOL_LIMITS.get(dataset_type)
    if limit is None:
        return
    num_samples = int(dataset_cfg.get('num_samples', 0) or 0)
    val_samples = int(dataset_cfg.get('val_samples', 0) or 0)
    assert num_samples >= 0, f'{path}: dataset.num_samples must be non-negative'
    assert val_samples >= 0, f'{path}: dataset.val_samples must be non-negative'
    assert num_samples + val_samples <= limit, (
        f'{path}: dataset.num_samples + dataset.val_samples must be <= {limit} '
        f'for {dataset_type}, got {num_samples} + {val_samples}'
    )


def test_template_yamls_parse_and_validate_splits():
    for path in sorted(TEMPLATES_ROOT.glob('*/*.yaml')):
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert isinstance(raw, dict), f'{path}: top-level YAML must be a mapping'
        _validate_dataset_split(raw, path)


def test_repository_templates_materialize_supported_models_without_value_errors():
    for path in sorted(TEMPLATES_ROOT.glob('*/*.yaml')):
        cfg = load_flex_config(path, [])
        _validate_dataset_split(cfg, path)

        model_cfg = cfg.get('model', {})
        if 'conv_layers' in model_cfg:
            continue

        backend = cfg.get('engine', {}).get('backend') or cfg.get('backend', {}).get('type')
        if backend == 'cuda':
            continue

        model = build_model(model_cfg, cfg['dataset']['input_shape'])
        assert model.inferred_shapes[-1] == (cfg['dataset']['num_classes'],)
