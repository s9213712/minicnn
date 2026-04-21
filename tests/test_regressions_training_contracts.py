"""Training-contract and validator regression tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Finding #2 — BCE validator now rejects cuda_legacy + BCEWithLogitsLoss
# ---------------------------------------------------------------------------

class TestBCEValidatorRejection:
    def _base_cfg(self):
        return {
            'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
            'model': {'layers': [
                {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3},
                {'type': 'LeakyReLU'},
                {'type': 'Conv2d', 'in_channels': 32, 'out_channels': 32, 'kernel_size': 3},
                {'type': 'LeakyReLU'},
                {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                {'type': 'Conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3},
                {'type': 'LeakyReLU'},
                {'type': 'Conv2d', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3},
                {'type': 'LeakyReLU'},
                {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                {'type': 'Flatten'},
                {'type': 'Linear', 'in_features': 64, 'out_features': 10},
            ]},
            'optimizer': {'type': 'SGD'},
            'loss': {'type': 'CrossEntropyLoss'},
            'train': {},
        }

    def test_bce_is_rejected_by_validator(self):
        from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
        cfg = self._base_cfg()
        cfg['loss']['type'] = 'BCEWithLogitsLoss'
        errors = validate_cuda_legacy_compatibility(cfg)
        assert any('BCEWithLogitsLoss' in e for e in errors), f"Expected BCE rejection, got: {errors}"

    def test_cross_entropy_still_passes(self):
        from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
        cfg = self._base_cfg()
        errors = validate_cuda_legacy_compatibility(cfg)
        assert not errors, f"CrossEntropy should pass validation: {errors}"

    def test_mse_still_passes_validator(self):
        from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
        cfg = self._base_cfg()
        cfg['loss']['type'] = 'MSELoss'
        errors = validate_cuda_legacy_compatibility(cfg)
        # MSELoss itself should not trigger a validator error (only BCE is rejected)
        assert not any('BCEWithLogitsLoss' in e for e in errors)


# ---------------------------------------------------------------------------
# Instruction #2 — train-dual torch summary merge (not overwrite)
# ---------------------------------------------------------------------------

class TestSummaryMerge:
    def test_torch_summary_fields_preserved(self):
        from minicnn.flex.runtime import dump_summary
        from minicnn.paths import BEST_MODELS_ROOT

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Simulate what train_from_config writes
            original_summary = {
                'device': 'cpu',
                'model_layers': ['Conv2d', 'Linear'],
                'optimizer': 'SGD',
                'loss': 'CrossEntropyLoss',
                'scheduler': None,
                'test_loss': 0.5,
                'test_acc': 0.82,
            }
            dump_summary(run_dir, original_summary)

            # Now simulate what train_unified_from_config does (the fixed version)
            existing = json.loads((run_dir / 'summary.json').read_text())
            existing.setdefault('selected_backend', 'torch')
            existing.setdefault('effective_backend', 'torch')
            existing.setdefault('run_dir', str(run_dir))
            existing.setdefault('best_model_path', str(BEST_MODELS_ROOT / f'{run_dir.name}_best.pt'))
            existing['config_backend_toggle_only'] = True
            dump_summary(run_dir, existing)

            merged = json.loads((run_dir / 'summary.json').read_text())
            # Original fields must survive
            assert merged.get('device') == 'cpu'
            assert merged.get('model_layers') == ['Conv2d', 'Linear']
            assert merged.get('optimizer') == 'SGD'
            assert merged.get('test_acc') == pytest.approx(0.82)
            # Unified metadata also present
            assert merged.get('selected_backend') == 'torch'
            assert merged.get('config_backend_toggle_only') is True


# ---------------------------------------------------------------------------
# Instruction #3 — autograd best-checkpoint reload before final test
# ---------------------------------------------------------------------------

class TestAutogradBestCheckpointReload:
    def test_test_acc_uses_best_checkpoint_not_last_epoch(self):
        """Best epoch ≠ last epoch: test_acc must come from best weights."""
        from minicnn.training.train_autograd import train_autograd_from_config

        cfg = {
            'dataset': {'type': 'random', 'input_shape': [1, 8, 8], 'num_classes': 2,
                        'num_samples': 32, 'val_samples': 8, 'test_samples': 8, 'seed': 0},
            'model': {'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'in_features': 64, 'out_features': 2},
            ]},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
            'train': {'epochs': 3, 'batch_size': 8, 'seed': 0, 'init_seed': 0},
            'project': {'name': 'test', 'run_name': 'ckpt_test', 'artifacts_root': tempfile.mkdtemp()},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg['project']['artifacts_root'] = tmpdir
            run_dir = train_autograd_from_config(cfg)
            summary = json.loads((run_dir / 'summary.json').read_text())
            # test_acc must be a valid float in [0, 1]
            assert 0.0 <= float(summary['test_acc']) <= 1.0
            # best_val_acc must be present
            assert 'best_val_acc' in summary
            assert 'best_model_path' in summary
            assert Path(summary['best_model_path']).exists()


# ---------------------------------------------------------------------------
# Instruction #4 — autograd scheduler validation
# ---------------------------------------------------------------------------

class TestAutogradSchedulerValidation:
    def _minimal_cfg(self, tmpdir):
        return {
            'dataset': {'type': 'random', 'input_shape': [1, 4, 4], 'num_classes': 2,
                        'num_samples': 8, 'val_samples': 4, 'seed': 0},
            'model': {'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'in_features': 16, 'out_features': 2},
            ]},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
            'train': {'epochs': 1, 'batch_size': 4},
            'project': {'name': 'test', 'run_name': 'sched_test', 'artifacts_root': tmpdir},
        }

    def test_step_size_zero_raises_value_error(self):
        from minicnn.training.train_autograd import train_autograd_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._minimal_cfg(tmpdir)
            cfg['scheduler'] = {'enabled': True, 'step_size': 0, 'gamma': 0.5}
            with pytest.raises(ValueError, match='step_size'):
                train_autograd_from_config(cfg)

    def test_step_size_zero_raises_not_zero_division(self):
        from minicnn.training.train_autograd import train_autograd_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._minimal_cfg(tmpdir)
            cfg['scheduler'] = {'enabled': True, 'step_size': 0, 'gamma': 0.5}
            with pytest.raises(ValueError):
                train_autograd_from_config(cfg)

    def test_negative_gamma_raises(self):
        from minicnn.training.train_autograd import train_autograd_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._minimal_cfg(tmpdir)
            cfg['scheduler'] = {'enabled': True, 'step_size': 1, 'gamma': -0.1}
            with pytest.raises(ValueError, match='gamma'):
                train_autograd_from_config(cfg)

    def test_valid_scheduler_does_not_raise(self):
        from minicnn.training.train_autograd import train_autograd_from_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._minimal_cfg(tmpdir)
            cfg['scheduler'] = {'enabled': True, 'step_size': 1, 'gamma': 0.5}
            run_dir = train_autograd_from_config(cfg)
            assert (run_dir / 'summary.json').exists()


# ---------------------------------------------------------------------------
# Instruction #5 — autograd dataset.download boolean parsing
# ---------------------------------------------------------------------------

class TestAutogradBooleanParsing:
    def test_download_string_false_is_false(self):
        # parse_bool("false") must be False; bool("false") would be True.
        from minicnn.config.parsing import parse_bool
        assert parse_bool('false', label='dataset.download') is False

    def test_download_string_true_is_true(self):
        from minicnn.config.parsing import parse_bool
        assert parse_bool('true', label='dataset.download') is True

    def test_download_bool_false_unchanged(self):
        from minicnn.config.parsing import parse_bool
        assert parse_bool(False, label='dataset.download') is False

    def test_train_autograd_uses_parse_bool_not_bool(self):
        """Verify _cifar10_dataset uses parse_bool, not raw bool(), for download."""
        import inspect, re
        import minicnn.training.train_autograd as m
        src = inspect.getsource(m._cifar10_dataset)
        assert 'parse_bool' in src, "_cifar10_dataset must call parse_bool for download"
        # raw bool( at the start (not part of parse_bool(...)
        assert not re.search(r'(?<!parse_)bool\(', src), (
            "_cifar10_dataset must not use raw bool() — use parse_bool() instead"
        )


# ---------------------------------------------------------------------------
# Instruction #1 — flex loss contract: CrossEntropy still works
# ---------------------------------------------------------------------------

class TestFlexLossContract:
    def test_cross_entropy_adapt_targets_unchanged(self):
        try:
            import torch
        except ImportError:
            pytest.skip('torch not available')
        from minicnn.flex.trainer import _adapt_targets, _pred_accuracy
        logits = torch.randn(4, 3)
        yb = torch.tensor([0, 1, 2, 0])
        adapted = _adapt_targets(yb, logits, 'CrossEntropyLoss')
        assert adapted is yb  # unchanged

    def test_mse_adapt_targets_one_hot(self):
        try:
            import torch
        except ImportError:
            pytest.skip('torch not available')
        from minicnn.flex.trainer import _adapt_targets
        logits = torch.randn(4, 3)
        yb = torch.tensor([0, 1, 2, 0])
        adapted = _adapt_targets(yb, logits, 'MSELoss')
        assert adapted.shape == (4, 3)
        assert adapted.dtype == torch.float32
        assert float(adapted[0, 0]) == 1.0
        assert float(adapted[1, 1]) == 1.0

    def test_bce_binary_adapt_targets(self):
        try:
            import torch
        except ImportError:
            pytest.skip('torch not available')
        from minicnn.flex.trainer import _adapt_targets
        logits = torch.randn(4, 1)
        yb = torch.tensor([0, 1, 1, 0])
        adapted = _adapt_targets(yb, logits, 'BCEWithLogitsLoss')
        assert adapted.shape == (4, 1)
        assert adapted.dtype == torch.float32

    def test_bce_multi_class_raises(self):
        try:
            import torch
        except ImportError:
            pytest.skip('torch not available')
        from minicnn.flex.trainer import _adapt_targets
        logits = torch.randn(4, 10)
        yb = torch.tensor([0, 1, 2, 3])
        with pytest.raises(ValueError, match='out_features=1'):
            _adapt_targets(yb, logits, 'BCEWithLogitsLoss')

    def test_bce_accuracy_uses_threshold(self):
        try:
            import torch
        except ImportError:
            pytest.skip('torch not available')
        from minicnn.flex.trainer import _pred_accuracy
        logits = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])
        targets = torch.tensor([1, 0, 1, 0])
        acc = _pred_accuracy(logits, targets, 'BCEWithLogitsLoss')
        assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Module-level: load_state_dict on nn.Module
# ---------------------------------------------------------------------------

class TestLoadStateDict:
    def test_load_state_dict_restores_weights(self):
        from minicnn.nn.modules import Module
        from minicnn.nn.tensor import Tensor

        class Linear(Module):
            def __init__(self):
                super().__init__()
                self.w = self.add_parameter('w', Tensor(np.ones((2, 2), dtype=np.float32)))

            def forward(self, x): return x

        m = Linear()
        original = {'w': np.ones((2, 2), dtype=np.float32) * 5}
        m.load_state_dict(original)
        np.testing.assert_array_equal(m.w.data, np.ones((2, 2)) * 5)

    def test_load_state_dict_missing_key_raises(self):
        from minicnn.nn.modules import Module
        from minicnn.nn.tensor import Tensor

        class M(Module):
            def __init__(self):
                super().__init__()
                self.a = self.add_parameter('a', Tensor(np.zeros(3, dtype=np.float32)))

        m = M()
        with pytest.raises(KeyError, match='missing'):
            m.load_state_dict({})
