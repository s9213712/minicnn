"""Phase 9: Augmentation config tests."""
from __future__ import annotations
import pytest


def test_flex_config_has_augmentation_defaults():
    from minicnn.flex.config import DEFAULT_CONFIG, load_flex_config
    assert 'augmentation' in DEFAULT_CONFIG
    aug = DEFAULT_CONFIG['augmentation']
    assert 'normalize' in aug
    assert 'random_crop' in aug
    assert 'horizontal_flip' in aug


def test_augmentation_config_round_trip():
    from minicnn.flex.config import load_flex_config
    cfg = load_flex_config(overrides=[
        'augmentation.random_crop=true',
        'augmentation.horizontal_flip=true',
        'augmentation.random_crop_padding=2',
    ])
    aug = cfg['augmentation']
    assert aug['random_crop'] is True
    assert aug['horizontal_flip'] is True
    assert int(aug['random_crop_padding']) == 2


def test_create_dataloaders_with_augmentation_cfg():
    pytest.importorskip('torch')
    import numpy as np
    from minicnn.flex.data import create_dataloaders
    dataset_cfg = {
        'type': 'random',
        'num_samples': 8,
        'val_samples': 4,
        'num_classes': 2,
        'input_shape': [3, 8, 8],
        'seed': 0,
    }
    train_cfg = {'batch_size': 4, 'num_workers': 0, 'seed': 0}
    aug_cfg = {'random_crop': True, 'random_crop_padding': 2, 'horizontal_flip': True}
    train_loader, val_loader = create_dataloaders(dataset_cfg, train_cfg, augmentation_cfg=aug_cfg)
    x, y = next(iter(train_loader))
    assert x.shape[-2:] == (8, 8)
