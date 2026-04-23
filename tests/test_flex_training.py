from pathlib import Path
import json

import pytest

from minicnn.flex.trainer import train_from_config
from minicnn.paths import BEST_MODELS_ROOT


def test_train_from_random_config(tmp_path: Path):
    cfg = {
        'project': {'name': 'test', 'run_name': 'pytest', 'artifacts_root': str(tmp_path)},
        'dataset': {'type': 'random', 'input_shape': [1, 8, 8], 'num_classes': 3, 'num_samples': 64, 'val_samples': 16, 'seed': 1},
        'model': {'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 16},
            {'type': 'ReLU'},
            {'type': 'Linear', 'out_features': 3},
        ]},
        'train': {'epochs': 1, 'batch_size': 16, 'device': 'cpu', 'amp': False, 'grad_accum_steps': 1, 'num_workers': 0},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.0},
        'scheduler': {'enabled': False},
        'runtime': {'save_every_n_epochs': 1},
    }
    run_dir = train_from_config(cfg)
    assert (BEST_MODELS_ROOT / f'{run_dir.name}_best.pt').exists()
    assert (BEST_MODELS_ROOT / f'{run_dir.name}_epoch_1.pt').exists()
    assert (run_dir / 'metrics.jsonl').exists()
    assert (run_dir / 'summary.json').exists()
    assert 'epoch_time_s' in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8')
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    assert summary['schema_version'] == 1
    assert summary['artifact_kind'] == 'training_run_summary'
    assert summary['status'] == 'ok'
    assert summary['selected_backend'] == 'torch'
    assert summary['effective_backend'] == 'torch'
    assert summary['variant'] == 'cpu'
    assert isinstance(summary['periodic_checkpoints'], list)
    assert 'test_loss' in summary
    assert 'test_acc' in summary


def test_optimizer_weight_decay_excludes_bias_and_norm_parameters():
    import torch

    from minicnn.flex.trainer import _optimizer_params

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Linear(4, 2),
    )
    cfg = {'type': 'SGD', 'lr': 0.01, 'weight_decay': 0.1}

    groups = _optimizer_params(model, cfg)

    assert len(groups) == 2
    assert groups[0]['weight_decay'] == 0.1
    assert groups[1]['weight_decay'] == 0.0
    assert len(groups[0]['params']) == 2
    assert len(groups[1]['params']) == 4
    assert 'weight_decay' not in cfg


def test_emit_training_event_formats_epoch_summary():
    from minicnn.flex.reporting import emit_training_event

    messages = []
    message = emit_training_event(
        'epoch_summary',
        {
            'epoch': 2,
            'epochs': 5,
            'train_metrics': {'loss': 1.2345, 'acc': 0.5},
            'val_metrics': {'loss': 0.9876, 'acc': 0.625},
            'lr': 0.001,
            'epoch_time_s': 3.2,
            'saved_best': True,
        },
        writer=messages.append,
    )

    assert message == messages[0]
    assert message.startswith('Epoch 2/5: loss=1.2345')
    assert 'val_acc=62.50%' in message
    assert message.endswith('saved_best')


def test_emit_training_event_formats_early_stop():
    from minicnn.flex.reporting import emit_training_event

    messages = []
    message = emit_training_event(
        'early_stop',
        {'epoch': 4, 'best_val_acc': 0.8125},
        writer=messages.append,
    )

    assert message == messages[0]
    assert message == 'Early stopping after 4 epochs; best val_acc=81.25%.'


def test_optimizer_type_override_drops_stale_default_kwargs():
    from minicnn.flex.config import load_flex_config
    from minicnn.flex.builder import build_optimizer
    import torch

    cfg = load_flex_config(None, ['optimizer.type=Adam', 'optimizer.lr=0.001'])
    assert cfg['optimizer'] == {'type': 'Adam', 'lr': 0.001}
    param = torch.nn.Parameter(torch.zeros(1))
    build_optimizer([param], cfg['optimizer'])


def test_override_parser_handles_nested_lists():
    from minicnn.flex.config import load_flex_config

    cfg = load_flex_config(None, ['dataset.patch_sizes=[3, [5, 7], wide value]'])

    assert cfg['dataset']['patch_sizes'] == [3, [5, 7], 'wide value']


def test_override_parser_updates_layer_list_index():
    from minicnn.flex.config import load_flex_config

    cfg = load_flex_config(None, ['model.layers.1.out_features=7'])

    assert cfg['model']['layers'][1]['out_features'] == 7


def test_optimizer_ignores_cuda_legacy_lr_fields_for_torch():
    from minicnn.flex.builder import build_optimizer
    import torch

    cfg = {
        'type': 'SGD',
        'lr': 0.01,
        'lr_conv1': 0.02,
        'lr_conv': 0.03,
        'lr_fc': 0.04,
        'grad_clip_global': 1.0,
        'momentum': 0.0,
    }
    param = torch.nn.Parameter(torch.zeros(1))

    optimizer = build_optimizer([param], cfg)

    assert optimizer.param_groups[0]['lr'] == 0.01


def test_train_loader_supports_cifar_style_augmentation():
    from minicnn.flex.data import create_dataloaders

    dataset_cfg = {
        'type': 'random',
        'input_shape': [3, 8, 8],
        'num_classes': 3,
        'num_samples': 8,
        'val_samples': 4,
        'seed': 1,
        'random_crop_padding': 2,
        'horizontal_flip': True,
    }
    train_cfg = {'batch_size': 4, 'num_workers': 0}

    train_loader, _ = create_dataloaders(dataset_cfg, train_cfg)
    xb, yb = next(iter(train_loader))

    assert tuple(xb.shape) == (4, 3, 8, 8)
    assert tuple(yb.shape) == (4,)


def test_train_loader_parses_string_false_for_horizontal_flip():
    from minicnn.flex.data import create_dataloaders

    dataset_cfg = {
        'type': 'random',
        'input_shape': [3, 8, 8],
        'num_classes': 3,
        'num_samples': 8,
        'val_samples': 4,
        'seed': 1,
        'horizontal_flip': 'false',
    }
    train_loader, _ = create_dataloaders(dataset_cfg, {'batch_size': 4, 'num_workers': 0})

    assert train_loader.dataset.horizontal_flip is False


def test_custom_dataset_factory_is_supported():
    from minicnn.flex.data import create_dataloaders, create_test_dataloader

    dataset_cfg = {
        'type': 'minicnn.extensions.custom_datasets:checkerboard_dataset',
        'input_shape': [1, 8, 8],
        'num_classes': 2,
        'num_samples': 8,
        'val_samples': 4,
        'test_samples': 4,
        'seed': 3,
    }
    train_cfg = {'batch_size': 4, 'num_workers': 0, 'seed': 5}

    train_loader, val_loader = create_dataloaders(dataset_cfg, train_cfg)
    test_loader = create_test_dataloader(dataset_cfg, train_cfg)

    xb, yb = next(iter(train_loader))
    xv, yv = next(iter(val_loader))
    xt, yt = next(iter(test_loader))

    assert tuple(xb.shape) == (4, 1, 8, 8)
    assert tuple(xv.shape) == (4, 1, 8, 8)
    assert tuple(xt.shape) == (4, 1, 8, 8)
    assert set(yb.tolist()) <= {0, 1}
    assert set(yv.tolist()) <= {0, 1}
    assert set(yt.tolist()) <= {0, 1}


def test_custom_dataset_factory_requires_dotted_import_path():
    from minicnn.flex.data import create_dataloaders

    dataset_cfg = {
        'type': 'checkerboard_dataset',
        'input_shape': [1, 8, 8],
        'num_classes': 2,
        'num_samples': 8,
        'val_samples': 4,
    }

    with pytest.raises(ValueError, match='package.module:factory'):
        create_dataloaders(dataset_cfg, {'batch_size': 4, 'num_workers': 0})


def test_create_dataloaders_reports_broken_torch_import(monkeypatch):
    import minicnn.flex.data as data

    monkeypatch.setattr(data, 'torch', None)
    monkeypatch.setattr(data, 'DataLoader', None)
    monkeypatch.setattr(data, 'AugmentedTensorDataset', None)
    monkeypatch.setattr(data, '_TORCH_IMPORT_ERROR', ImportError('broken torch install for test'))

    dataset_cfg = {
        'type': 'random',
        'input_shape': [1, 8, 8],
        'num_classes': 2,
        'num_samples': 8,
        'val_samples': 4,
    }

    with pytest.raises(RuntimeError, match='broken torch install for test'):
        data.create_dataloaders(dataset_cfg, {'batch_size': 4, 'num_workers': 0})


def test_create_dataloaders_reports_missing_torch_with_install_hint(monkeypatch):
    import minicnn.flex.data as data

    monkeypatch.setattr(data, 'torch', None)
    monkeypatch.setattr(data, 'DataLoader', None)
    monkeypatch.setattr(data, 'AugmentedTensorDataset', None)
    monkeypatch.setattr(data, '_TORCH_IMPORT_ERROR', None)

    dataset_cfg = {
        'type': 'random',
        'input_shape': [1, 8, 8],
        'num_classes': 2,
        'num_samples': 8,
        'val_samples': 4,
    }

    with pytest.raises(RuntimeError, match='train-flex requires PyTorch to load flex datasets'):
        data.create_dataloaders(dataset_cfg, {'batch_size': 4, 'num_workers': 0})


def test_dataset_loader_registry_still_supports_random_dataset():
    import minicnn.flex.data as data

    x_train, y_train, x_val, y_val, x_test, y_test = data._load_dataset_arrays(
        {
            'type': 'random',
            'input_shape': [1, 4, 4],
            'num_classes': 2,
            'num_samples': 4,
            'val_samples': 2,
            'seed': 1,
        },
        {'batch_size': 2},
    )

    assert x_train.shape == (4, 1, 4, 4)
    assert y_train.shape == (4,)
    assert x_val.shape == (2, 1, 4, 4)
    assert y_val.shape == (2,)
    assert x_test is None
    assert y_test is None


def test_train_from_config_applies_init_seed_before_model_build(tmp_path: Path, monkeypatch):
    import torch
    import minicnn.flex.trainer as trainer

    observed_seeds = []

    def build_seed_probe(_model_cfg, input_shape):
        observed_seeds.append(torch.initial_seed())
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(int(input_shape[0] * input_shape[1] * input_shape[2]), 2),
        )

    monkeypatch.setattr(trainer, 'build_model', build_seed_probe)
    cfg = {
        'project': {'name': 'test', 'run_name': 'pytest-init-seed', 'artifacts_root': str(tmp_path)},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 4, 4],
            'num_classes': 2,
            'num_samples': 4,
            'val_samples': 2,
            'seed': 1,
        },
        'model': {'layers': [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 2}]},
        'train': {
            'epochs': 1,
            'batch_size': 2,
            'device': 'cpu',
            'amp': False,
            'grad_accum_steps': 1,
            'num_workers': 0,
            'init_seed': 123,
        },
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.0},
        'scheduler': {'enabled': False},
    }

    train_from_config(cfg)

    assert observed_seeds == [123]


def test_augmented_dataset_changes_random_seed_by_epoch():
    import torch

    from minicnn.flex.data import AugmentedTensorDataset

    x = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    y = torch.zeros((1,), dtype=torch.long)
    dataset = AugmentedTensorDataset(x, y, random_crop_padding=2, horizontal_flip=True, seed=123)

    seed_epoch0 = dataset._generator(0).initial_seed()
    dataset.set_epoch(1)
    seed_epoch1 = dataset._generator(0).initial_seed()

    assert seed_epoch1 != seed_epoch0


def test_grad_accumulation_flushes_final_partial_window(tmp_path: Path, monkeypatch):
    import torch
    import minicnn.flex.trainer as trainer

    steps = []

    class CountingSGD(torch.optim.SGD):
        def step(self, *args, **kwargs):
            steps.append(1)
            return super().step(*args, **kwargs)

    monkeypatch.setattr(
        trainer,
        'build_optimizer',
        lambda params, _cfg: CountingSGD(params, lr=0.01, momentum=0.0),
    )
    cfg = {
        'project': {'name': 'test', 'run_name': 'pytest-flush', 'artifacts_root': str(tmp_path)},
        'dataset': {'type': 'random', 'input_shape': [1, 4, 4], 'num_classes': 2, 'num_samples': 5, 'val_samples': 2, 'seed': 1},
        'model': {'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ]},
        'train': {'epochs': 1, 'batch_size': 2, 'device': 'cpu', 'amp': False, 'grad_accum_steps': 4, 'num_workers': 0},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.0},
        'scheduler': {'enabled': False},
    }

    train_from_config(cfg)

    assert len(steps) == 1
