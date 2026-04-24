from __future__ import annotations

import json
import warnings


def test_cuda_native_artifact_schema_is_explicit(tmp_path):
    from minicnn.unified.trainer import train_unified_from_config

    cfg = {
        'engine': {
            'backend': 'cuda_native',
            'planner_strategy': 'reuse',
        },
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 11,
        },
        'model': {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        'train': {
            'batch_size': 2,
            'epochs': 1,
            'init_seed': 11,
        },
        'optimizer': {
            'type': 'SGD',
            'lr': 0.01,
        },
        'loss': {
            'type': 'CrossEntropyLoss',
        },
        'project': {
            'artifacts_root': str(tmp_path),
        },
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_dir = train_unified_from_config(cfg)

    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    row = json.loads((run_dir / 'metrics.jsonl').read_text(encoding='utf-8').strip().splitlines()[-1])

    assert summary['schema_name'] == 'minicnn.cuda_native.training.summary'
    assert summary['schema_version'] == 1
    assert summary['artifact_kind'] == 'training_run_summary'
    assert summary['execution_mode'] == 'reference_numpy'
    assert summary['effective_execution_mode'] == 'reference_numpy'
    assert summary['tensor_execution_device'] == 'cpu'
    assert summary['device_runtime']['execution_mode'] == 'reference_numpy'
    assert summary['device_runtime']['tensor_execution_device'] == 'cpu'
    assert summary['device_runtime']['reserved_buffer_count'] >= 1
    assert summary['device_runtime']['reserved_bytes'] >= 1
    assert summary['support_tier_assessment']['highest_tier'] == 'stable'
    assert summary['checkpoint_contract']['format'] == 'npz'
    assert summary['checkpoint_contract']['version'] == 1
    assert summary['checkpoint_contract']['best_model_path_key'] == 'best_model_path'

    assert row['schema_name'] == 'minicnn.cuda_native.training.metrics.epoch'
    assert row['schema_version'] == 1
    assert row['artifact_kind'] == 'training_metrics_epoch'
    assert row['execution_mode'] == 'reference_numpy'
    assert row['effective_execution_mode'] == 'reference_numpy'
    assert row['tensor_execution_device'] == 'cpu'
    assert row['device_runtime']['execution_mode'] == 'reference_numpy'
    assert row['device_runtime']['tensor_execution_device'] == 'cpu'
    assert row['device_runtime']['reserved_buffer_count'] >= 1
    assert row['device_runtime']['reserved_bytes'] >= 1
    assert row['support_tier_assessment']['highest_tier'] == 'stable'
