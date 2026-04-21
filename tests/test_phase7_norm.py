"""Phase 7: Normalization formalization regression tests."""
from __future__ import annotations
import pytest


def test_builder_batchnorm2d_in_autograd_model():
    from minicnn.models.builder import build_model_from_config
    from minicnn.nn.layers import BatchNorm2d, Conv2d
    cfg = {
        'layers': [
            {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3},
            {'type': 'BatchNorm2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ]
    }
    model, shape = build_model_from_config(cfg, input_shape=(1, 8, 8))
    assert shape == (2,)
    modules = list(model)
    assert isinstance(modules[1], BatchNorm2d)


def test_cuda_legacy_rejects_layernorm():
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3},
            {'type': 'LayerNorm'},
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
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('LayerNorm' in e for e in errors)


def test_cuda_legacy_rejects_groupnorm():
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3},
            {'type': 'GroupNorm'},
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
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('GroupNorm' in e for e in errors)
