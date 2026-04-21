"""P0/P1 regression tests: BCE binary label rules and cuda_legacy capability."""
from __future__ import annotations
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# P0: flex BCE binary label rules
# ---------------------------------------------------------------------------

def _flex_bce_cfg(labels, out_features=1):
    """Minimal flex train config for BCE binary path."""
    return {
        'dataset': {
            'type': 'random',
            'num_samples': len(labels),
            'val_samples': 2,
            'num_classes': 2,
            'input_shape': [2],
        },
        'train': {'epochs': 1, 'batch_size': len(labels)},
        'model': {'layers': [
            {'type': 'Linear', 'in_features': 2, 'out_features': out_features},
        ]},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'BCEWithLogitsLoss'},
    }


def test_flex_bce_binary_accepts_legal_labels():
    pytest.importorskip('torch')
    import torch
    from minicnn.flex.trainer import _adapt_targets
    logits = torch.zeros(4, 1)
    yb = torch.tensor([0, 1, 0, 1])
    out = _adapt_targets(yb, logits, 'BCEWithLogitsLoss')
    assert out.shape == (4, 1)


def test_flex_bce_binary_rejects_label_2():
    pytest.importorskip('torch')
    import torch
    from minicnn.flex.trainer import _adapt_targets
    logits = torch.zeros(4, 1)
    yb = torch.tensor([0, 1, 2, 0])
    with pytest.raises(ValueError, match=r'\{0, 1\}'):
        _adapt_targets(yb, logits, 'BCEWithLogitsLoss')


def test_flex_bce_binary_rejects_label_9():
    pytest.importorskip('torch')
    import torch
    from minicnn.flex.trainer import _adapt_targets
    logits = torch.zeros(3, 1)
    yb = torch.tensor([0, 9, 1])
    with pytest.raises(ValueError, match='CrossEntropyLoss'):
        _adapt_targets(yb, logits, 'BCEWithLogitsLoss')


def test_flex_bce_multioutput_still_rejected():
    """Out_features > 1 with BCE still raises (out-shape check comes first)."""
    pytest.importorskip('torch')
    import torch
    from minicnn.flex.trainer import _adapt_targets
    logits = torch.zeros(4, 10)
    yb = torch.tensor([0, 1, 0, 1])
    with pytest.raises(ValueError, match='out_features=1'):
        _adapt_targets(yb, logits, 'BCEWithLogitsLoss')


# ---------------------------------------------------------------------------
# P0: autograd BCE binary label rules
# ---------------------------------------------------------------------------

def test_autograd_bce_binary_accepts_legal_labels():
    from minicnn.training.train_autograd import _dense_targets
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    result = _dense_targets(labels, (4, 1), 'BCEWithLogitsLoss')
    assert result.shape == (4, 1)
    assert set(result.flatten().tolist()) <= {0.0, 1.0}


def test_autograd_bce_binary_rejects_label_2():
    from minicnn.training.train_autograd import _dense_targets
    labels = np.array([0, 2, 1], dtype=np.int64)
    with pytest.raises(ValueError, match=r'\{0, 1\}'):
        _dense_targets(labels, (3, 1), 'BCEWithLogitsLoss')


def test_autograd_bce_binary_rejects_label_9():
    from minicnn.training.train_autograd import _dense_targets
    labels = np.array([0, 9, 1], dtype=np.int64)
    with pytest.raises(ValueError, match='CrossEntropyLoss'):
        _dense_targets(labels, (3, 1), 'BCEWithLogitsLoss')


def test_autograd_bce_random_10class_fails_fast():
    """End-to-end: random dataset with num_classes=10, BCE, out=1 must fail fast."""
    from minicnn.training.train_autograd import train_autograd_from_config
    cfg = {
        'dataset': {'type': 'random', 'num_samples': 8, 'val_samples': 4,
                    'num_classes': 10, 'input_shape': [4]},
        'train': {'epochs': 1, 'batch_size': 8},
        'model': {'layers': [
            {'type': 'Linear', 'out_features': 1},
        ]},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'BCEWithLogitsLoss'},
    }
    with pytest.raises(ValueError, match=r'\{0, 1\}'):
        train_autograd_from_config(cfg)


def test_autograd_mse_dense_unaffected():
    """MSE multi-output path must not be broken by BCE check."""
    from minicnn.training.train_autograd import _dense_targets
    labels = np.array([0, 5, 9], dtype=np.int64)
    result = _dense_targets(labels, (3, 10), 'MSELoss')
    assert result.shape == (3, 10)
    assert result[0, 0] == 1.0
    assert result[1, 5] == 1.0


# ---------------------------------------------------------------------------
# P1: cuda_legacy capability declaration
# ---------------------------------------------------------------------------

def test_cuda_legacy_supported_loss_excludes_bce():
    from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED
    assert 'BCEWithLogitsLoss' not in CUDA_LEGACY_SUPPORTED['loss']
    assert 'CrossEntropyLoss' in CUDA_LEGACY_SUPPORTED['loss']
    assert 'MSELoss' in CUDA_LEGACY_SUPPORTED['loss']


def test_cuda_legacy_validator_rejects_bce_with_clear_message():
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
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'BCEWithLogitsLoss'},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('BCEWithLogitsLoss' in e for e in errors)
    assert any('CrossEntropyLoss' in e for e in errors)


def test_cuda_legacy_validator_accepts_crossentropy():
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
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    loss_errors = [e for e in errors if 'loss' in e.lower() or 'BCE' in e or 'CrossEntropy' in e]
    assert loss_errors == []
