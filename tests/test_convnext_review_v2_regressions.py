from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_config_rejects_unknown_backend(capsys):
    from minicnn.cli import main

    rc = main([
        'validate-config',
        '--config',
        'templates/cifar10/convnext_like.yaml',
        'engine.backend=bogus',
        '--format',
        'json',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload['ok'] is False
    assert any("Unknown backend 'bogus'" in err for err in payload['errors'])


def test_convnext_snake_case_aliases_build_and_run_forward():
    import torch

    from minicnn.flex.builder import build_model

    model = build_model({
        'layers': [
            {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 1},
            {'type': 'depthwise_conv2d', 'kernel_size': 3},
            {'type': 'layernorm2d'},
            {'type': 'pointwise_conv2d', 'out_channels': 16},
            {'type': 'GELU'},
            {'type': 'pointwise_conv2d', 'out_channels': 8},
            {'type': 'convnext_block'},
            {'type': 'GlobalAvgPool2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]
    }, input_shape=(3, 8, 8))

    y = model(torch.randn(2, 3, 8, 8))
    assert tuple(y.shape) == (2, 10)


def test_train_flex_rejects_backend_mismatch_without_creating_artifact_run(tmp_path, capsys):
    from minicnn.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main([
            'train-flex',
            '--config',
            'configs/flex_cnn.yaml',
            'engine.backend=cuda_native',
            'dataset.num_samples=2',
            'dataset.val_samples=1',
            'train.epochs=1',
            'train.batch_size=1',
            'train.device=cpu',
            f'project.artifacts_root={tmp_path}',
        ])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "train-flex cannot run engine.backend='cuda_native'" in captured.err
    assert not any(tmp_path.iterdir())


def test_smoke_reports_missing_cifar_files_not_just_directory_existence(tmp_path, monkeypatch):
    import minicnn._cli_readonly as cli_readonly
    import minicnn.framework.health as health

    data_root = tmp_path / 'data' / 'cifar-10-batches-py'
    data_root.mkdir(parents=True)
    monkeypatch.setattr(cli_readonly, 'DATA_ROOT', data_root)
    monkeypatch.setattr(health, 'DATA_ROOT', data_root)

    payload = cli_readonly.run_smoke_checks()
    cifar_check = next(check for check in payload['checks'] if check['name'] == 'cifar10_data')

    assert cifar_check['ok'] is False
    assert cifar_check['severity'] == 'warning'
    assert 'data_batch_1' in cifar_check['details']['missing']


def test_show_model_counts_explicit_convnext_parameterized_layers():
    from minicnn.introspection.model_view import build_model_view_from_config

    cfg = {
        'dataset': {'input_shape': [3, 32, 32]},
        'model': {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 1},
                {'type': 'PointwiseConv2d', 'out_channels': 16},
                {'type': 'LayerNorm2d'},
                {'type': 'convnext_block'},
            ],
        },
    }
    view = build_model_view_from_config(cfg)
    assert view.summary['parameterized_layers'] == 4


def test_optimizer_params_excludes_layernorm2d_weight_from_weight_decay():
    from minicnn.flex.builder import build_model
    from minicnn.flex.trainer import _optimizer_params

    model = build_model({
        'layers': [
            {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 3, 'padding': 1},
            {'type': 'LayerNorm2d'},
            {'type': 'GlobalAvgPool2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ],
    }, input_shape=(3, 8, 8))
    groups = _optimizer_params(model, {'type': 'AdamW', 'lr': 1e-3, 'weight_decay': 0.05})

    assert isinstance(groups, list)
    no_decay_group = next(group for group in groups if group['weight_decay'] == 0.0)
    layernorm_weight = model[1].weight
    assert any(param is layernorm_weight for param in no_decay_group['params'])


def test_convnext_tiny_respects_model_num_classes():
    import torch

    from minicnn.flex.builder import build_model

    model = build_model({'name': 'convnext_tiny', 'num_classes': 5}, input_shape=(3, 32, 32))
    y = model(torch.randn(2, 3, 32, 32))
    assert tuple(y.shape) == (2, 5)


def test_train_flex_does_not_leave_run_dir_when_dataset_setup_fails(tmp_path):
    from minicnn.flex.config import load_flex_config
    from minicnn.flex.trainer import train_from_config

    cfg = load_flex_config(REPO_ROOT / 'templates/cifar10/convnext_explicit.yaml', [])
    cfg['dataset']['data_root'] = str(tmp_path / 'missing-cifar')
    cfg['dataset']['download'] = False
    cfg['dataset']['num_samples'] = 8
    cfg['dataset']['val_samples'] = 4
    cfg['train']['epochs'] = 1
    cfg['train']['batch_size'] = 2
    cfg['train']['device'] = 'cpu'
    cfg['project']['artifacts_root'] = str(tmp_path / 'artifacts')

    with pytest.raises(FileNotFoundError):
        train_from_config(cfg)

    artifacts_root = Path(cfg['project']['artifacts_root'])
    assert not artifacts_root.exists() or not any(artifacts_root.iterdir())


def test_random_dataset_rejects_negative_split_sizes():
    from minicnn.flex._datasets import _random_dataset

    with pytest.raises(ValueError) as excinfo:
        _random_dataset({'type': 'random', 'num_samples': -1, 'val_samples': 4}, {})

    assert 'Dataset split invalid' in str(excinfo.value)


def test_convnext_explicit_template_docs_residual_boundary_is_honest():
    template_path = REPO_ROOT / 'templates/cifar10/convnext_explicit.yaml'
    raw = template_path.read_text(encoding='utf-8')
    assert 'does NOT include residual add or layer scale' in raw

    smoke_path = REPO_ROOT / 'templates/cifar10/convnext_explicit_smoke.yaml'
    smoke_raw = smoke_path.read_text(encoding='utf-8')
    assert 'does not encode residual add or layer scale semantics' in smoke_raw
