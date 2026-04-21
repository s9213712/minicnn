from __future__ import annotations

import pytest
from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility


def _valid_cuda_cfg(**overrides):
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'ReLU'},
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'ReLU'},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        'train': {'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
    }
    cfg.update(overrides)
    return cfg


def test_cuda_legacy_rejects_batchnorm2d():
    cfg = _valid_cuda_cfg()
    cfg['model']['layers'].insert(1, {'type': 'BatchNorm2d'})
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('BatchNorm2d' in e for e in errors)


def test_cuda_legacy_rejects_avgpool2d():
    cfg = _valid_cuda_cfg()
    layers = cfg['model']['layers']
    layers[4] = {'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 2}
    errors = validate_cuda_legacy_compatibility(cfg)
    assert errors


def test_cuda_legacy_rejects_non_3_kernel():
    cfg = _valid_cuda_cfg()
    cfg['model']['layers'][0] = {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'padding': 0}
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('kernel_size=3' in e for e in errors)


def test_cuda_legacy_rejects_amp_true():
    cfg = _valid_cuda_cfg()
    cfg['train']['amp'] = True
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('amp' in e for e in errors)


def test_cuda_legacy_rejects_bce_with_logits_loss():
    cfg = _valid_cuda_cfg()
    cfg['loss'] = {'type': 'BCEWithLogitsLoss'}
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('BCEWithLogitsLoss' in e or 'bce' in e.lower() for e in errors)


def test_flex_builder_relu():
    from minicnn.flex.builder import build_model
    cfg = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 8},
            {'type': 'ReLU'},
            {'type': 'Linear', 'out_features': 2},
        ]
    }
    model = build_model(cfg, input_shape=(16,))
    import torch
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 2)


def test_flex_builder_silu():
    from minicnn.flex.builder import build_model
    cfg = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 8},
            {'type': 'SiLU'},
            {'type': 'Linear', 'out_features': 2},
        ]
    }
    model = build_model(cfg, input_shape=(16,))
    import torch
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 2)


def test_flex_builder_tanh():
    from minicnn.flex.builder import build_model
    cfg = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 8},
            {'type': 'Tanh'},
            {'type': 'Linear', 'out_features': 2},
        ]
    }
    model = build_model(cfg, input_shape=(16,))
    import torch
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 2)


def test_autograd_sgd_works():
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    import numpy as np
    p = Parameter(np.array([1.0, 2.0], dtype=np.float32))
    p.grad = np.array([0.1, 0.2], dtype=np.float32)
    opt = SGD([p], lr=0.1)
    opt.step()
    assert p.data[0] < 1.0


def test_autograd_adam_works():
    from minicnn.optim.adam import Adam
    from minicnn.nn.tensor import Parameter
    import numpy as np
    p = Parameter(np.array([1.0, 2.0], dtype=np.float32))
    p.grad = np.array([0.1, 0.2], dtype=np.float32)
    opt = Adam([p], lr=0.01)
    opt.step()
    assert p.data[0] < 1.0
