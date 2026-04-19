from pathlib import Path

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
    }
    run_dir = train_from_config(cfg)
    assert (BEST_MODELS_ROOT / f'{run_dir.name}_best.pt').exists()
    assert (run_dir / 'metrics.jsonl').exists()
    assert (run_dir / 'summary.json').exists()
    assert 'epoch_time_s' in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8')


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


def test_optimizer_ignores_cuda_legacy_lr_fields_for_torch():
    from minicnn.flex.builder import build_optimizer
    import torch

    cfg = {
        'type': 'SGD',
        'lr': 0.01,
        'lr_conv1': 0.02,
        'lr_conv': 0.03,
        'lr_fc': 0.04,
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
