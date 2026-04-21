from __future__ import annotations

import numpy as np
import pytest
from minicnn.nn.tensor import Parameter
from minicnn.optim.adamw import AdamW
from minicnn.optim.rmsprop import RMSprop


def _make_param(val):
    p = Parameter(np.array(val, dtype=np.float32))
    return p


def test_adamw_decoupled_weight_decay():
    p = _make_param([1.0, 2.0])
    p.grad = np.array([0.0, 0.0], dtype=np.float32)
    opt = AdamW([p], lr=0.01, weight_decay=0.1)
    data_before = p.data.copy()
    opt.step()
    assert np.all(p.data < data_before), "Weight decay should reduce param values even with zero gradient"


def test_adamw_weight_decay_not_in_gradient():
    p1 = _make_param([1.0])
    p1.grad = np.array([0.5], dtype=np.float32)

    p2 = _make_param([1.0])
    p2.grad = np.array([0.5 + 0.1 * 1.0], dtype=np.float32)

    opt_adamw = AdamW([p1], lr=0.01, weight_decay=0.1)
    from minicnn.optim.adam import Adam
    opt_adam = Adam([p2], lr=0.01, weight_decay=0.0)

    opt_adamw.step()
    opt_adam.step()

    assert not np.allclose(p1.data, p2.data, atol=1e-6), "AdamW and Adam+WD should differ (decoupled vs coupled)"


def test_adamw_step_returns_dict():
    p = _make_param([1.0])
    p.grad = np.array([0.1], dtype=np.float32)
    opt = AdamW([p], lr=0.001)
    result = opt.step()
    assert 'updated' in result
    assert result['updated'] == 1


def test_adamw_skips_none_grad():
    p = _make_param([1.0])
    opt = AdamW([p], lr=0.01)
    result = opt.step()
    assert result['updated'] == 0


def test_rmsprop_v_cache_evolves():
    p = _make_param([1.0, -1.0])
    p.grad = np.array([2.0, 3.0], dtype=np.float32)
    opt = RMSprop([p], lr=0.01, alpha=0.9, eps=1e-8)
    assert np.all(opt.v[0] == 0.0)
    opt.step()
    expected_v = 0.1 * np.array([4.0, 9.0], dtype=np.float32)
    np.testing.assert_allclose(opt.v[0], expected_v, rtol=1e-6)


def test_rmsprop_update_correct():
    p = _make_param([1.0])
    grad = np.array([1.0], dtype=np.float32)
    p.grad = grad.copy()
    opt = RMSprop([p], lr=0.01, alpha=0.0, eps=0.0)
    data_before = p.data.copy()
    opt.step()
    np.testing.assert_allclose(p.data, data_before - 0.01 * 1.0 / (np.sqrt(1.0) + 0.0), rtol=1e-5)


def test_train_autograd_factory_adamw():
    from minicnn.training.train_autograd import _make_optimizer
    from minicnn.nn.tensor import Parameter
    import numpy as np
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = _make_optimizer(params, {'type': 'AdamW', 'lr': 0.001})
    assert isinstance(opt, AdamW)


def test_train_autograd_factory_rmsprop():
    from minicnn.training.train_autograd import _make_optimizer
    from minicnn.nn.tensor import Parameter
    import numpy as np
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = _make_optimizer(params, {'type': 'RMSprop', 'lr': 0.01})
    assert isinstance(opt, RMSprop)


def test_train_autograd_factory_adamw_lowercase():
    from minicnn.training.train_autograd import _make_optimizer
    from minicnn.nn.tensor import Parameter
    import numpy as np
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = _make_optimizer(params, {'type': 'adamw', 'lr': 0.001})
    assert isinstance(opt, AdamW)


def test_cuda_legacy_still_rejects_adamw():
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
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
        'optimizer': {'type': 'AdamW', 'lr': 0.001},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('AdamW' in e or 'SGD or Adam' in e or 'optimizer' in e.lower() for e in errors)


def test_cuda_legacy_still_rejects_rmsprop():
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
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
        'optimizer': {'type': 'RMSprop', 'lr': 0.01},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('RMSprop' in e or 'SGD or Adam' in e or 'optimizer' in e.lower() for e in errors)
