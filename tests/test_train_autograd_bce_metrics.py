"""Regression tests for train_autograd BCE/CE accuracy and input validation."""
from __future__ import annotations

import numpy as np
import pytest

from minicnn.training.train_autograd import _accuracy, train_autograd_from_config
from minicnn.models import build_model_from_config
from minicnn.nn import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_model_cfg(out: int):
    return {'layers': [{'type': 'Linear', 'out_features': out}]}


def _run_train(epochs: int = 1, loss_type: str = 'CrossEntropyLoss', num_classes: int = 2):
    cfg = {
        'dataset': {
            'type': 'random',
            'input_shape': [4],
            'num_classes': num_classes,
            'num_samples': 16,
            'val_samples': 4,
            'test_samples': 4,
        },
        'model': _linear_model_cfg(num_classes if loss_type != 'BCEWithLogitsLoss' else 1),
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': loss_type},
        'train': {'epochs': epochs, 'batch_size': 8},
    }
    return train_autograd_from_config(cfg)


# ---------------------------------------------------------------------------
# B1 — BCE binary accuracy uses threshold, not argmax
# ---------------------------------------------------------------------------

class TestBCEBinaryAccuracy:
    def test_threshold_predicts_positive_for_large_logit(self):
        """logit >> 0 should predict class 1, not always 0 (argmax bug)."""
        rng = np.random.default_rng(0)

        class ConstantModel:
            def __call__(self, x):
                # always return large positive logit → should predict 1
                return Tensor(np.full((x.data.shape[0], 1), 5.0, dtype=np.float32))
            def train(self, mode=True):
                pass

        x = rng.normal(size=(10, 4)).astype(np.float32)
        y = np.ones(10, dtype=np.int64)
        acc = _accuracy(ConstantModel(), x, y, batch_size=10, loss_type='BCEWithLogitsLoss')
        assert acc == 1.0, f"expected 1.0, got {acc}"

    def test_threshold_predicts_negative_for_negative_logit(self):
        rng = np.random.default_rng(1)

        class ConstantModel:
            def __call__(self, x):
                return Tensor(np.full((x.data.shape[0], 1), -5.0, dtype=np.float32))
            def train(self, mode=True):
                pass

        x = rng.normal(size=(10, 4)).astype(np.float32)
        y = np.zeros(10, dtype=np.int64)
        acc = _accuracy(ConstantModel(), x, y, batch_size=10, loss_type='BCEWithLogitsLoss')
        assert acc == 1.0, f"expected 1.0, got {acc}"

    def test_argmax_would_have_been_wrong(self):
        """Verify argmax on (N,1) always returns 0 — confirms we needed the fix."""
        logits = np.array([[5.0], [3.0], [2.0]], dtype=np.float32)
        argmax_pred = logits.argmax(axis=1)
        threshold_pred = (logits[:, 0] >= 0.0).astype(np.int64)
        assert (argmax_pred == 0).all(), "argmax should always be 0 for (N,1)"
        assert (threshold_pred == 1).all(), "threshold should be 1 for positive logits"

    def test_bce_training_runs_without_error(self):
        run_dir = _run_train(epochs=1, loss_type='BCEWithLogitsLoss', num_classes=2)
        assert run_dir.exists()


# ---------------------------------------------------------------------------
# CrossEntropy accuracy still correct
# ---------------------------------------------------------------------------

class TestCrossEntropyAccuracy:
    def test_argmax_used_for_multiclass(self):
        rng = np.random.default_rng(2)

        class FixedModel:
            def __call__(self, x):
                n = x.data.shape[0]
                logits = np.zeros((n, 3), dtype=np.float32)
                logits[:, 1] = 10.0  # always predicts class 1
                return Tensor(logits)
            def train(self, mode=True):
                pass

        x = rng.normal(size=(8, 4)).astype(np.float32)
        y = np.ones(8, dtype=np.int64)  # all class 1
        acc = _accuracy(FixedModel(), x, y, batch_size=8, loss_type='CrossEntropyLoss')
        assert acc == 1.0

    def test_crossentropy_training_runs(self):
        run_dir = _run_train(epochs=1, loss_type='CrossEntropyLoss', num_classes=3)
        assert run_dir.exists()


# ---------------------------------------------------------------------------
# C1 — unknown optimizer raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_optimizer_raises():
    cfg = {
        'dataset': {'type': 'random', 'input_shape': [4], 'num_classes': 2, 'num_samples': 8, 'val_samples': 4},
        'model': _linear_model_cfg(2),
        'optimizer': {'type': 'RMSProp', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'train': {'epochs': 1, 'batch_size': 4},
    }
    with pytest.raises(ValueError, match='unsupported optimizer'):
        train_autograd_from_config(cfg)


# ---------------------------------------------------------------------------
# C2 — epochs < 1 raises ValueError
# ---------------------------------------------------------------------------

def test_epochs_zero_raises():
    cfg = {
        'dataset': {'type': 'random', 'input_shape': [4], 'num_classes': 2, 'num_samples': 8, 'val_samples': 4},
        'model': _linear_model_cfg(2),
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'train': {'epochs': 0, 'batch_size': 4},
    }
    with pytest.raises(ValueError, match='epochs'):
        train_autograd_from_config(cfg)
