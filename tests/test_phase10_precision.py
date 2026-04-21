"""Phase 10: Precision / dtype policy tests."""
from __future__ import annotations
import pytest


def test_autograd_rejects_amp():
    from minicnn.training.train_autograd import train_autograd_from_config
    cfg = {
        'dataset': {'type': 'random', 'num_samples': 4, 'val_samples': 2,
                    'input_shape': [1, 4, 4], 'num_classes': 2},
        'train': {'epochs': 1, 'batch_size': 2, 'amp': True},
        'model': {'layers': [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 2}]},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
    }
    with pytest.raises(ValueError, match='amp'):
        train_autograd_from_config(cfg)


def test_cuda_legacy_rejects_amp():
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        'engine': {'backend': 'cuda_legacy'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'train': {'amp': True},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('amp' in e.lower() for e in errors)


def test_autograd_fp32_default_works():
    """Autograd train with no amp config should succeed (fp32 default)."""
    from minicnn.training.train_autograd import train_autograd_from_config
    cfg = {
        'dataset': {'type': 'random', 'num_samples': 4, 'val_samples': 2,
                    'input_shape': [1, 4, 4], 'num_classes': 2},
        'train': {'epochs': 1, 'batch_size': 2},
        'model': {'layers': [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 2}]},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
    }
    path = train_autograd_from_config(cfg)
    assert path is not None
