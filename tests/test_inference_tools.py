from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml


torch = pytest.importorskip('torch')
PIL = pytest.importorskip('PIL.Image')


def _write_config(tmp_path: Path) -> tuple[dict, Path]:
    cfg = {
        'project': {'name': 'minicnn', 'run_name': 'toy', 'artifacts_root': str(tmp_path / 'artifacts')},
        'dataset': {
            'type': 'cifar10',
            'input_shape': [3, 32, 32],
            'num_classes': 3,
            'class_names': ['zero', 'one', 'two'],
        },
        'model': {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 3},
            ],
        },
        'train': {
            'batch_size': 2,
            'device': 'cpu',
        },
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'scheduler': {'enabled': False},
    }
    config_path = tmp_path / 'toy_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return cfg, config_path


def _write_checkpoint(tmp_path: Path, cfg: dict) -> Path:
    from minicnn.flex.builder import build_model

    model = build_model(cfg['model'], input_shape=cfg['dataset']['input_shape'])
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model[1].bias.copy_(torch.tensor([0.0, 8.0, -8.0], dtype=torch.float32))
    checkpoint_path = tmp_path / 'toy_best.pt'
    torch.save({'model_state': model.state_dict()}, checkpoint_path)
    return checkpoint_path


def test_evaluate_checkpoint_accepts_custom_test_npz(tmp_path):
    from minicnn.inference import evaluate_checkpoint

    cfg, _config_path = _write_config(tmp_path)
    checkpoint_path = _write_checkpoint(tmp_path, cfg)
    test_data_path = tmp_path / 'test_data.npz'
    x = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    y = np.ones((4,), dtype=np.int64)
    np.savez(test_data_path, x=x, y=y)

    result = evaluate_checkpoint(
        cfg,
        checkpoint_path=checkpoint_path,
        device_name='cpu',
        batch_size=2,
        test_data_path=str(test_data_path),
    )

    assert result['schema_version'] == 1
    assert result['kind'] == 'checkpoint_evaluation'
    assert result['status'] == 'ok'
    assert result['dataset_source'] == str(test_data_path)
    assert result['num_samples'] == 4
    assert result['accuracy'] == pytest.approx(1.0)


def test_predict_image_resizes_and_returns_topk(tmp_path):
    from PIL import Image

    from minicnn.inference import predict_image

    cfg, _config_path = _write_config(tmp_path)
    checkpoint_path = _write_checkpoint(tmp_path, cfg)
    image_path = tmp_path / 'sample.png'
    Image.new('RGB', (96, 48), color=(120, 30, 10)).save(image_path)

    result = predict_image(
        cfg,
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        device_name='cpu',
        topk=2,
    )

    assert result['status'] == 'ok'
    assert result['preprocessing']['original_size'] == [96, 48]
    assert result['preprocessing']['target_size'] == [32, 32]
    assert [entry['label'] for entry in result['predictions']] == ['one', 'zero']


def test_load_best_model_path_from_summary_rejects_non_training_summary(tmp_path):
    from minicnn.inference import load_best_model_path_from_summary

    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps({
        'schema_version': 1,
        'artifact_kind': 'project_info',
        'best_model_path': 'best.pt',
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='training_run_summary'):
        load_best_model_path_from_summary(summary_path)


def test_load_best_model_path_from_summary_rejects_unknown_schema_version(tmp_path):
    from minicnn.inference import load_best_model_path_from_summary

    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps({
        'schema_version': 99,
        'artifact_kind': 'training_run_summary',
        'best_model_path': 'best.pt',
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='schema_version'):
        load_best_model_path_from_summary(summary_path)


def test_load_best_model_path_from_summary_accepts_extra_cuda_native_metadata(tmp_path):
    from minicnn.inference import load_best_model_path_from_summary

    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps({
        'schema_name': 'minicnn.cuda_native.training.summary',
        'schema_version': 1,
        'artifact_kind': 'training_run_summary',
        'best_model_path': '/tmp/best.npz',
        'checkpoint_contract': {
            'format': 'npz',
            'version': 1,
            'best_model_path_key': 'best_model_path',
        },
    }), encoding='utf-8')

    assert load_best_model_path_from_summary(summary_path) == '/tmp/best.npz'


def test_load_best_model_path_from_summary_keeps_backward_compat_without_schema_name(tmp_path):
    from minicnn.inference import load_best_model_path_from_summary

    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps({
        'schema_version': 1,
        'artifact_kind': 'training_run_summary',
        'best_model_path': 'legacy_best.npz',
    }), encoding='utf-8')

    assert load_best_model_path_from_summary(summary_path) == 'legacy_best.npz'


def test_cli_evaluate_checkpoint_outputs_structured_json(tmp_path, capsys):
    from minicnn.cli import main

    _cfg, config_path = _write_config(tmp_path)
    checkpoint_path = _write_checkpoint(tmp_path, _cfg)
    test_data_path = tmp_path / 'test_data.npz'
    np.savez(
        test_data_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.ones((2,), dtype=np.int64),
    )

    rc = main([
        'evaluate-checkpoint',
        '--config',
        str(config_path),
        '--checkpoint',
        str(checkpoint_path),
        '--test-data',
        str(test_data_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload['command'] == 'evaluate-checkpoint'
    assert payload['schema_version'] == 1
    assert payload['kind'] == 'checkpoint_evaluation'
    assert payload['status'] == 'ok'
    assert payload['accuracy'] == pytest.approx(1.0)
