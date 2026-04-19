from pathlib import Path

from minicnn.flex.trainer import train_from_config


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
    assert (run_dir / 'best.pt').exists()
    assert (run_dir / 'metrics.jsonl').exists()
    assert (run_dir / 'summary.json').exists()


def test_optimizer_type_override_drops_stale_default_kwargs():
    from minicnn.flex.config import load_flex_config
    from minicnn.flex.builder import build_optimizer
    import torch

    cfg = load_flex_config(None, ['optimizer.type=Adam', 'optimizer.lr=0.001'])
    assert cfg['optimizer'] == {'type': 'Adam', 'lr': 0.001}
    param = torch.nn.Parameter(torch.zeros(1))
    build_optimizer([param], cfg['optimizer'])


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
