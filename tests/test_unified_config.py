import json
import os
from pathlib import Path

from minicnn.unified.config import load_unified_config
from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility, compile_to_legacy_experiment


def test_unified_config_override_backend():
    cfg = load_unified_config(None, ['engine.backend=cuda_legacy', 'train.epochs=3'])
    assert cfg['engine']['backend'] == 'cuda_legacy'
    assert cfg['train']['epochs'] == 3


def test_unified_config_override_supports_layer_list_index():
    cfg = load_unified_config(None, ['model.layers.1.out_features=7'])

    assert cfg['model']['layers'][1]['out_features'] == 7


def test_unified_config_type_override_clears_stale_optimizer_fields():
    cfg = load_unified_config(None, ['optimizer.type=Adam'])

    assert cfg['optimizer'] == {'type': 'Adam'}


def test_unified_config_sorts_type_override_before_other_fields():
    cfg = load_unified_config(None, ['optimizer.lr=0.001', 'optimizer.type=Adam'])

    assert cfg['optimizer'] == {'type': 'Adam', 'lr': 0.001}


def test_compile_supported_dual_config():
    cfg = {
        'project': {'name': 'x', 'run_name': 'y', 'artifacts_root': 'artifacts'},
        'engine': {'backend': 'cuda_legacy'},
        'runtime': {'cuda_variant': 'handmade'},
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10, 'num_samples': 128, 'val_samples': 32, 'seed': 42},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        'train': {'epochs': 2, 'batch_size': 16, 'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005},
    }
    assert validate_cuda_legacy_compatibility(cfg) == []
    exp = compile_to_legacy_experiment(cfg)
    # Verify correct ModelConfig fields are set (not the old dead-write c1_out etc.)
    assert exp.model.c_in == 3
    assert exp.model.conv_layers[0]['out_c'] == 32
    assert exp.model.conv_layers[1]['out_c'] == 32
    assert exp.model.conv_layers[2]['out_c'] == 64
    assert exp.model.conv_layers[3]['out_c'] == 64
    assert exp.model.conv_layers[1]['pool'] is True
    assert exp.model.conv_layers[3]['pool'] is True
    assert exp.optim.lr_fc == 0.005


def test_compile_cuda_legacy_maps_distinct_dataset_init_and_train_seeds():
    cfg = load_unified_config(Path(__file__).resolve().parents[1] / 'configs' / 'dual_backend_cnn.yaml', [
        'engine.backend=cuda_legacy',
        'dataset.seed=111',
        'train.init_seed=222',
        'train.seed=333',
    ])

    exp = compile_to_legacy_experiment(cfg)

    assert exp.train.dataset_seed == 111
    assert exp.train.init_seed == 222
    assert exp.train.train_seed == 333


def test_compile_cuda_legacy_maps_global_grad_clip():
    cfg = load_unified_config(Path(__file__).resolve().parents[1] / 'configs' / 'dual_backend_cnn.yaml', [
        'engine.backend=cuda_legacy',
        'optimizer.grad_clip_global=2.5',
    ])

    assert validate_cuda_legacy_compatibility(cfg) == []
    exp = compile_to_legacy_experiment(cfg)

    assert exp.optim.grad_clip_global == 2.5


def test_cuda_legacy_accepts_native_variant_runtime_option():
    cfg = load_unified_config(None, ['engine.backend=cuda_legacy', 'runtime.cuda_variant=handmade'])
    errors = validate_cuda_legacy_compatibility(cfg)
    assert all('runtime.cuda_variant' not in err for err in errors)


def test_cuda_legacy_runtime_env_restores_previous_values(monkeypatch):
    import minicnn.unified.trainer as trainer

    trainer._MANAGED_CUDA_ENV.clear()
    monkeypatch.setenv('MINICNN_CUDA_SO', '/tmp/external.so')
    summary = {}

    trainer._configure_cuda_legacy_runtime({'runtime': {'cuda_so': 'cpp/custom.so'}}, summary)
    assert os.environ['MINICNN_CUDA_SO'] == 'cpp/custom.so'

    trainer._configure_cuda_legacy_runtime({'runtime': {}}, {})
    assert os.environ['MINICNN_CUDA_SO'] == '/tmp/external.so'


def test_cuda_legacy_summary_preserves_returned_test_acc(tmp_path, monkeypatch):
    import minicnn.config.settings as settings
    import minicnn.training.train_cuda as train_cuda
    import minicnn.unified.trainer as trainer
    from minicnn.config.schema import ExperimentConfig

    run_dir = tmp_path / 'cuda-run'
    run_dir.mkdir()
    cfg = load_unified_config(None, ['engine.backend=cuda_legacy'])

    monkeypatch.setattr(trainer, 'create_run_dir', lambda _cfg: run_dir)
    monkeypatch.setattr(trainer, 'compile_to_legacy_experiment', lambda _cfg: ExperimentConfig())
    monkeypatch.setattr(trainer, 'summarize_legacy_mapping', lambda _cfg: {'backend': 'cuda_legacy'})
    monkeypatch.setattr(trainer, '_reload_legacy_modules_after_config', lambda: None)
    monkeypatch.setattr(settings, 'apply_experiment_config', lambda _exp: None)
    monkeypatch.setattr(train_cuda, 'main', lambda: {'test_acc': 12.5, 'best_model_path': 'best.npz'})

    result = trainer.train_unified_from_config(cfg)
    summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))

    assert summary['schema_version'] == 1
    assert summary['artifact_kind'] == 'training_run_summary'
    assert summary['status'] == 'ok'
    assert summary['selected_backend'] == 'cuda_legacy'
    assert summary['effective_backend'] == 'cuda_legacy'
    assert summary['periodic_checkpoints'] == []
    assert summary['test_acc'] == 12.5
    assert summary['test_loss'] is None
    assert summary['best_model_path'] == 'best.npz'
