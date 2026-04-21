from __future__ import annotations

import math
import numpy as np
import pytest
from minicnn.schedulers.step import StepLR
from minicnn.schedulers.cosine import CosineAnnealingLR


class _FakeOptimizer:
    def __init__(self, lr):
        self.lr = lr


def test_steplr_reduces_at_correct_epoch():
    opt = _FakeOptimizer(0.1)
    sched = StepLR(opt, step_size=3, gamma=0.1)
    lrs = []
    for _ in range(6):
        sched.step()
        lrs.append(opt.lr)
    assert abs(lrs[2] - 0.01) < 1e-7, f"Expected 0.01 at epoch 3, got {lrs[2]}"
    assert abs(lrs[5] - 0.001) < 1e-7, f"Expected 0.001 at epoch 6, got {lrs[5]}"
    assert abs(lrs[1] - 0.1) < 1e-7, f"lr should not change at epoch 2, got {lrs[1]}"


def test_steplr_min_lr():
    opt = _FakeOptimizer(0.1)
    sched = StepLR(opt, step_size=1, gamma=0.1, min_lr=0.05)
    sched.step()
    assert opt.lr == pytest.approx(0.05, rel=1e-6)
    sched.step()
    assert opt.lr == pytest.approx(0.05, rel=1e-6)


def test_cosine_lr_follows_curve():
    T_max = 10
    lr_max = 0.1
    lr_min = 0.0
    opt = _FakeOptimizer(lr_max)
    sched = CosineAnnealingLR(opt, T_max=T_max, lr_min=lr_min)
    for epoch in range(1, T_max + 1):
        sched.step()
        expected = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * epoch / T_max))
        assert abs(opt.lr - expected) < 1e-6, f"epoch {epoch}: expected {expected}, got {opt.lr}"


def test_cosine_lr_at_T_max_is_lr_min():
    opt = _FakeOptimizer(0.1)
    sched = CosineAnnealingLR(opt, T_max=5, lr_min=0.0)
    for _ in range(5):
        sched.step()
    assert abs(opt.lr) < 1e-6


def test_train_autograd_factory_none():
    from minicnn.training.train_autograd import _make_scheduler
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = SGD(params, lr=0.01)
    sched = _make_scheduler(opt, {'enabled': False})
    assert sched is None


def test_train_autograd_factory_none_type():
    from minicnn.training.train_autograd import _make_scheduler
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = SGD(params, lr=0.01)
    sched = _make_scheduler(opt, {'enabled': True, 'type': 'none'})
    assert sched is None


def test_train_autograd_factory_step():
    from minicnn.training.train_autograd import _make_scheduler
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = SGD(params, lr=0.01)
    sched = _make_scheduler(opt, {'enabled': True, 'type': 'step', 'step_size': 5})
    assert isinstance(sched, StepLR)


def test_train_autograd_factory_cosine():
    from minicnn.training.train_autograd import _make_scheduler
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = SGD(params, lr=0.01)
    sched = _make_scheduler(opt, {'enabled': True, 'type': 'cosine', 'T_max': 10})
    assert isinstance(sched, CosineAnnealingLR)


def test_train_autograd_factory_default_is_step():
    from minicnn.training.train_autograd import _make_scheduler
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    params = [Parameter(np.zeros(2, dtype=np.float32))]
    opt = SGD(params, lr=0.01)
    sched = _make_scheduler(opt, {'enabled': True})
    assert isinstance(sched, StepLR)
