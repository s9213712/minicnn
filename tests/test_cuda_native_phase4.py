"""Phase 4 MVP integration tests: CLI surface, trainer bridge, doctor, and config validation."""
from __future__ import annotations

import json
import pytest
import yaml


# ---------------------------------------------------------------------------
# CLI surface tests
# ---------------------------------------------------------------------------

class TestCLICapabilities:
    def test_cuda_native_capabilities_returns_dict(self):
        from minicnn.cuda_native.api import get_capability_summary
        caps = get_capability_summary()
        assert isinstance(caps, dict)

    def test_capability_summary_has_required_keys(self):
        from minicnn.cuda_native.api import get_capability_summary
        caps = get_capability_summary()
        assert 'experimental' in caps
        assert 'supported_ops' in caps
        assert 'forward' in caps or 'forward_only' in caps

    def test_capability_experimental_is_true(self):
        from minicnn.cuda_native.api import get_capability_summary
        caps = get_capability_summary()
        assert caps.get('experimental') is True

    def test_supported_ops_includes_core_ops(self):
        from minicnn.cuda_native.api import get_capability_summary
        caps = get_capability_summary()
        supported = caps.get('supported_ops', [])
        for op in ('BatchNorm2d', 'Conv2d', 'ReLU', 'Flatten', 'Linear'):
            assert op in supported, f'{op} missing from supported_ops'

    def test_unsupported_ops_listed(self):
        from minicnn.cuda_native.api import get_capability_summary
        caps = get_capability_summary()
        unsupported = caps.get('unsupported_ops', [])
        assert len(unsupported) > 0


# ---------------------------------------------------------------------------
# validate-cuda-native-config equivalent
# ---------------------------------------------------------------------------

class TestValidateCudaNativeConfig:
    def _minimal_cfg(self, layers):
        return {
            'engine': {'backend': 'cuda_native'},
            'dataset': {'type': 'random', 'input_shape': [1, 8, 8], 'num_classes': 2, 'num_samples': 4, 'val_samples': 2},
            'model': {'layers': layers},
            'train': {'batch_size': 2, 'epochs': 1},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
        }

    def test_valid_minimal_config_passes(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        errors = validate_cuda_native_config(cfg)
        assert errors == [], f'Unexpected errors: {errors}'

    def test_unsupported_op_rejected(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'GroupNorm'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        errors = validate_cuda_native_config(cfg)
        assert len(errors) > 0

    def test_missing_out_features_rejected(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear'},  # missing out_features
        ])
        errors = validate_cuda_native_config(cfg)
        assert len(errors) > 0

    def test_batchnorm2d_is_allowed_by_generic_validator(self):
        from minicnn.cuda_native.validators import validate_cuda_native_model_config
        cfg = self._minimal_cfg([
            {'type': 'BatchNorm2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        errors = validate_cuda_native_model_config(cfg['model'])
        assert errors == []

    def test_full_config_accepts_batchnorm2d_now_that_backward_exists(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'BatchNorm2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        errors = validate_cuda_native_config(cfg)
        assert errors == []

    def test_dataset_type_foo_rejected(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['dataset']['type'] = 'foo'
        errors = validate_cuda_native_config(cfg)
        assert any('dataset.type' in e for e in errors)

    def test_loss_bcewithlogits_rejected(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['loss']['type'] = 'BCEWithLogitsLoss'
        errors = validate_cuda_native_config(cfg)
        assert any('loss.type' in e for e in errors)

    def test_optimizer_adam_rejected(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['optimizer']['type'] = 'Adam'
        errors = validate_cuda_native_config(cfg)
        assert any('optimizer.type' in e for e in errors)

    def test_optimizer_momentum_is_allowed(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['optimizer']['momentum'] = 0.9
        errors = validate_cuda_native_config(cfg)
        assert errors == []

    def test_scheduler_enabled_step_is_allowed(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['scheduler'] = {'enabled': True, 'type': 'StepLR'}
        errors = validate_cuda_native_config(cfg)
        assert errors == []

    def test_scheduler_enabled_cosine_is_allowed(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['scheduler'] = {'enabled': True, 'type': 'CosineAnnealingLR', 'T_max': 5}
        errors = validate_cuda_native_config(cfg)
        assert errors == []

    def test_scheduler_enabled_plateau_is_allowed(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['scheduler'] = {'enabled': True, 'type': 'ReduceLROnPlateau', 'factor': 0.5, 'patience': 1}
        errors = validate_cuda_native_config(cfg)
        assert errors == []

    def test_amp_and_grad_accum_rejected(self):
        from minicnn.cuda_native.api import validate_cuda_native_config
        cfg = self._minimal_cfg([
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ])
        cfg['train']['amp'] = True
        cfg['train']['grad_accum_steps'] = 2
        errors = validate_cuda_native_config(cfg)
        assert any('train.amp=true' in e for e in errors)
        assert any('train.grad_accum_steps' in e for e in errors)


# ---------------------------------------------------------------------------
# Trainer bridge
# ---------------------------------------------------------------------------

class TestTrainerBridge:
    def _minimal_cfg(self):
        return {
            'engine': {'backend': 'cuda_native'},
            'dataset': {'type': 'random', 'input_shape': [1, 8, 8], 'num_classes': 2, 'num_samples': 8, 'val_samples': 4},
            'model': {'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ]},
            'train': {'batch_size': 4, 'epochs': 1},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
            'run': {'output_root': '/tmp/minicnn_test_phase4'},
        }

    def test_trainer_routes_cuda_native(self):
        import warnings
        from minicnn.unified.trainer import train_unified_from_config
        cfg = self._minimal_cfg()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_dir = train_unified_from_config(cfg)
        assert run_dir.exists()
        assert (run_dir / 'summary.json').exists()

    def test_trainer_summary_has_backend(self):
        import warnings
        from minicnn.unified.trainer import train_unified_from_config
        cfg = self._minimal_cfg()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_dir = train_unified_from_config(cfg)
        summary = json.loads((run_dir / 'summary.json').read_text())
        assert summary.get('selected_backend') == 'cuda_native'
        assert summary.get('effective_backend') == 'cuda_native'

    def test_unknown_backend_raises(self):
        from minicnn.unified.trainer import train_unified_from_config
        cfg = {'engine': {'backend': 'nonexistent'}}
        with pytest.raises(ValueError, match='nonexistent'):
            train_unified_from_config(cfg)

    def test_trainer_accepts_batchnorm2d_training_graph(self):
        import warnings
        from minicnn.unified.trainer import train_unified_from_config
        cfg = self._minimal_cfg()
        cfg['model']['layers'] = [
            {'type': 'BatchNorm2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_dir = train_unified_from_config(cfg)
        assert run_dir.exists()
        assert (run_dir / 'summary.json').exists()

    def test_trainer_rejects_unsupported_optimizer_before_runtime(self):
        import warnings
        from minicnn.unified.trainer import train_unified_from_config
        cfg = self._minimal_cfg()
        cfg['optimizer']['type'] = 'Adam'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='optimizer.type'):
                train_unified_from_config(cfg)

    def test_validate_cuda_native_cli_rejects_invalid_optimizer(self, tmp_path, capsys):
        from minicnn.cli import main

        cfg = {
            'engine': {'backend': 'cuda_native'},
            'dataset': {'type': 'random', 'input_shape': [1, 8, 8], 'num_classes': 2, 'num_samples': 4, 'val_samples': 2},
            'model': {'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ]},
            'train': {'batch_size': 2, 'epochs': 1},
            'optimizer': {'type': 'Adam', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
        }
        cfg['optimizer']['type'] = 'Adam'
        config_path = tmp_path / 'cuda_native_invalid.yaml'
        config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

        rc = main(['validate-cuda-native-config', '--config', str(config_path)])
        captured = capsys.readouterr()

        assert rc == 2
        assert 'optimizer.type' in captured.out

    def test_trainer_accepts_scheduler_and_records_it(self):
        import warnings
        from minicnn.unified.trainer import train_unified_from_config
        cfg = self._minimal_cfg()
        cfg['scheduler'] = {'enabled': True, 'type': 'StepLR', 'step_size': 1, 'gamma': 0.5}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_dir = train_unified_from_config(cfg)
        summary = json.loads((run_dir / 'summary.json').read_text())
        assert summary.get('scheduler') == 'StepLR'


# ---------------------------------------------------------------------------
# Doctor integration
# ---------------------------------------------------------------------------

class TestDoctorIntegration:
    def test_doctor_includes_cuda_native(self):
        from minicnn.framework.health import doctor
        result = doctor()
        assert 'cuda_native' in result

    def test_doctor_cuda_native_is_dict(self):
        from minicnn.framework.health import doctor
        result = doctor()
        assert isinstance(result['cuda_native'], dict)

    def test_doctor_cuda_native_has_experimental(self):
        from minicnn.framework.health import doctor
        result = doctor()
        assert result['cuda_native'].get('experimental') is True
