from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np


def _assert_minimum_artifact_contract(run_dir: Path) -> tuple[dict, dict]:
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    rows = (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').strip().splitlines()
    row = json.loads(rows[-1])

    assert summary['schema_name'] == 'minicnn.cuda_native.training.summary'
    assert summary['schema_version'] == 1
    assert summary['artifact_kind'] == 'training_run_summary'
    assert Path(summary['best_model_path']).exists()
    assert summary['checkpoint_contract']['format'] == 'npz'
    assert summary['checkpoint_contract']['version'] == 1
    assert 'planner' in summary

    assert row['schema_name'] == 'minicnn.cuda_native.training.metrics.epoch'
    assert row['schema_version'] == 1
    assert row['artifact_kind'] == 'training_metrics_epoch'
    assert 'epoch' in row
    assert 'train_loss' in row
    assert 'val_loss' in row
    assert 'val_acc' in row
    assert 'epoch_time_s' in row
    return summary, row


def _train(tmp_path: Path, cfg: dict) -> Path:
    from minicnn.unified.trainer import train_unified_from_config

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return train_unified_from_config(cfg)


def test_cuda_native_sequential_classifier_smoke(tmp_path):
    cfg = {
        'engine': {'backend': 'cuda_native', 'planner_strategy': 'reuse'},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 21,
        },
        'model': {'layers': [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 2}]},
        'train': {'batch_size': 2, 'epochs': 1, 'init_seed': 21},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'project': {'artifacts_root': str(tmp_path)},
    }

    run_dir = _train(tmp_path, cfg)
    _assert_minimum_artifact_contract(run_dir)


def test_cuda_native_ordered_dag_add_smoke(tmp_path):
    cfg = {
        'engine': {'backend': 'cuda_native', 'planner_strategy': 'reuse'},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 22,
        },
        'model': {
            'layers': [
                {'type': 'Identity', 'output': 'stem'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
                {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'merged'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        'train': {'batch_size': 2, 'epochs': 1, 'init_seed': 22},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'project': {'artifacts_root': str(tmp_path)},
    }

    run_dir = _train(tmp_path, cfg)
    _assert_minimum_artifact_contract(run_dir)


def test_cuda_native_ordered_dag_concat_smoke(tmp_path):
    cfg = {
        'engine': {'backend': 'cuda_native', 'planner_strategy': 'reuse'},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 23,
        },
        'model': {
            'layers': [
                {'type': 'Identity', 'output': 'stem'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
                {'type': 'Concat', 'inputs': ['left', 'right'], 'axis': 1, 'output': 'merged'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        'train': {'batch_size': 2, 'epochs': 1, 'init_seed': 23},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'project': {'artifacts_root': str(tmp_path)},
    }

    run_dir = _train(tmp_path, cfg)
    _assert_minimum_artifact_contract(run_dir)


def test_cuda_native_amp_grad_accum_smoke_and_checkpoint_compatibility(tmp_path):
    from minicnn.inference import load_best_model_path_from_summary

    cfg = {
        'engine': {'backend': 'cuda_native', 'planner_strategy': 'reuse'},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 1,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 24,
        },
        'model': {'layers': [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 1}]},
        'train': {
            'batch_size': 2,
            'epochs': 1,
            'amp': True,
            'amp_loss_scale': 128.0,
            'grad_accum_steps': 2,
            'init_seed': 24,
        },
        'optimizer': {'type': 'AdamW', 'lr': 0.01, 'weight_decay': 0.01},
        'loss': {'type': 'BCEWithLogitsLoss'},
        'project': {'artifacts_root': str(tmp_path)},
    }

    run_dir = _train(tmp_path, cfg)
    summary, row = _assert_minimum_artifact_contract(run_dir)

    assert row['amp']['enabled'] is True
    assert 'optimizer_runtime' in row
    assert 'planner' in row
    summary_path = run_dir / 'summary.json'
    resolved_best = load_best_model_path_from_summary(summary_path)
    assert Path(resolved_best) == Path(summary['best_model_path'])
    with np.load(Path(resolved_best)) as checkpoint:
        assert len(checkpoint.files) > 0


def test_cuda_native_fixed_seed_smoke_is_reproducible(tmp_path):
    cfg = {
        'engine': {'backend': 'cuda_native', 'planner_strategy': 'reuse'},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 31,
        },
        'model': {'layers': [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 2}]},
        'train': {'batch_size': 2, 'epochs': 1, 'init_seed': 31},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
    }

    run_dir_a = _train(tmp_path / 'run_a', {**cfg, 'project': {'artifacts_root': str(tmp_path / 'run_a')}})
    run_dir_b = _train(tmp_path / 'run_b', {**cfg, 'project': {'artifacts_root': str(tmp_path / 'run_b')}})

    summary_a, row_a = _assert_minimum_artifact_contract(run_dir_a)
    summary_b, row_b = _assert_minimum_artifact_contract(run_dir_b)

    assert summary_a['best_val_acc'] == summary_b['best_val_acc']
    assert row_a['train_loss'] == row_b['train_loss']
    assert row_a['val_loss'] == row_b['val_loss']
    assert row_a['val_acc'] == row_b['val_acc']

    with np.load(Path(summary_a['best_model_path'])) as ckpt_a, np.load(Path(summary_b['best_model_path'])) as ckpt_b:
        assert ckpt_a.files == ckpt_b.files
        for key in ckpt_a.files:
            np.testing.assert_array_equal(ckpt_a[key], ckpt_b[key])


def test_cuda_native_droppath_smoke(tmp_path):
    cfg = {
        'engine': {'backend': 'cuda_native', 'planner_strategy': 'reuse'},
        'dataset': {
            'type': 'random',
            'input_shape': [3, 8, 8],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 41,
        },
        'model': {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'padding': 1},
                {'type': 'DropPath', 'p': 0.2},
                {'type': 'GlobalAvgPool2d'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        'train': {'batch_size': 2, 'epochs': 1, 'init_seed': 41},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'project': {'artifacts_root': str(tmp_path)},
    }

    run_dir = _train(tmp_path, cfg)
    summary, _row = _assert_minimum_artifact_contract(run_dir)

    assert summary['support_tier_assessment']['highest_tier'] == 'beta'
    assert 'DropPath' in summary['support_tier_assessment']['ops_by_tier']['beta']
