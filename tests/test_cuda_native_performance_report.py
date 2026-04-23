from __future__ import annotations

import json
import warnings


def test_cuda_native_summary_and_metrics_include_performance_telemetry(tmp_path):
    from minicnn.unified.trainer import train_unified_from_config

    cfg = {
        'engine': {
            'backend': 'cuda_native',
            'planner_strategy': 'reuse',
        },
        'dataset': {
            'type': 'random',
            'input_shape': [1, 8, 8],
            'num_classes': 1,
            'num_samples': 8,
            'val_samples': 4,
            'seed': 7,
        },
        'model': {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 1},
            ],
        },
        'train': {
            'batch_size': 2,
            'epochs': 1,
            'amp': True,
            'amp_loss_scale': 128.0,
            'grad_accum_steps': 2,
            'init_seed': 7,
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 0.01,
            'weight_decay': 0.01,
        },
        'loss': {
            'type': 'BCEWithLogitsLoss',
        },
        'project': {
            'artifacts_root': str(tmp_path),
        },
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_dir = train_unified_from_config(cfg)

    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    metrics_lines = (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').strip().splitlines()
    row = json.loads(metrics_lines[-1])

    assert summary['amp'] is True
    assert 'amp_runtime' in summary
    assert 'optimizer_runtime' in summary
    assert 'performance_report' in summary
    assert summary['performance_report']['planner']['strategy'] == 'reuse'
    assert summary['performance_report']['training']['grad_accum_steps'] == 2
    assert summary['performance_report']['training']['amp_enabled'] is True
    assert summary['optimizer_runtime']['optimizer_type'] == 'adamw'
    assert summary['optimizer_runtime']['state_tensor_count'] > 0
    assert summary['optimizer_runtime']['state_total_bytes'] > 0
    assert 'cache_allocations' in summary['amp_runtime']

    assert row['amp']['enabled'] is True
    assert 'loss_scale' in row['amp']
    assert 'optimizer_runtime' in row
    assert row['optimizer_runtime']['optimizer_type'] == 'adamw'
    assert row['optimizer_runtime']['steps_epoch'] > 0
    assert row['optimizer_runtime']['state_tensor_count'] > 0
    assert 'state_tensor_allocations_epoch' in row['optimizer_runtime']
    assert 'state_tensor_updates_epoch' in row['optimizer_runtime']
    assert row['planner']['strategy'] == 'reuse'
    assert 'peak_live_bytes' in row['planner']
    assert 'reuse_events' in row['planner']
