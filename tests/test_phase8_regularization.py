"""Phase 8: Regularization tests."""
from __future__ import annotations
import numpy as np
import pytest


def test_builder_dropout_in_autograd_model():
    from minicnn.models.builder import build_model_from_config
    from minicnn.nn.layers import Dropout
    cfg = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Dropout', 'p': 0.5},
            {'type': 'Linear', 'out_features': 4},
        ]
    }
    model, shape = build_model_from_config(cfg, input_shape=(8,))
    assert shape == (4,)
    modules = list(model)
    assert isinstance(modules[1], Dropout)


def test_label_smoothing_raises_loss():
    """CE with label_smoothing > 0 should give higher loss for correct predictions."""
    from minicnn.nn.tensor import Tensor
    from minicnn.nn.tensor import cross_entropy
    # Perfect logits: class 0 has huge score
    logits = np.array([[10.0, -10.0, -10.0]], dtype=np.float32)
    targets = np.array([0], dtype=np.int64)
    loss_no_smooth = cross_entropy(Tensor(logits), targets, label_smoothing=0.0)
    loss_smooth = cross_entropy(Tensor(logits), targets, label_smoothing=0.1)
    assert loss_smooth.data > loss_no_smooth.data


def test_label_smoothing_backward():
    """label_smoothing backward should produce gradients."""
    from minicnn.nn.tensor import Tensor
    from minicnn.nn.tensor import cross_entropy
    logits = Tensor(np.array([[2.0, 1.0, 0.5]], dtype=np.float32), requires_grad=True)
    targets = np.array([0], dtype=np.int64)
    loss = cross_entropy(logits, targets, label_smoothing=0.1)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (1, 3)


def test_sgd_weight_decay_reduces_params():
    """SGD with weight_decay should decay parameters faster than without."""
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Tensor, Parameter
    p1 = Parameter(np.ones((4,), dtype=np.float32))
    p2 = Parameter(np.ones((4,), dtype=np.float32))
    grad = np.ones((4,), dtype=np.float32) * 0.0
    p1.grad = grad.copy()
    p2.grad = grad.copy()
    opt_decay = SGD([p1], lr=0.1, weight_decay=0.1)
    opt_no_decay = SGD([p2], lr=0.1, weight_decay=0.0)
    opt_decay.step()
    opt_no_decay.step()
    # With zero gradient, weight_decay should push p1 toward zero more than p2
    assert float(p1.data.sum()) < float(p2.data.sum())


def test_autograd_amp_rejection():
    """train_autograd must reject amp=true with a clear error."""
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
