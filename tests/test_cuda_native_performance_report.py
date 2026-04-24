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

    assert summary['schema_name'] == 'minicnn.cuda_native.training.summary'
    assert summary['schema_version'] == 1
    assert summary['amp'] is True
    assert 'amp_runtime' in summary
    assert 'optimizer_runtime' in summary
    assert 'performance_report' in summary
    assert summary['performance_report']['planner']['strategy'] == 'reuse'
    assert 'efficiency' in summary['performance_report']
    assert 'bottlenecks' in summary['performance_report']
    assert 'runtime' in summary['performance_report']
    assert summary['performance_report']['training']['grad_accum_steps'] == 2
    assert summary['performance_report']['training']['amp_enabled'] is True
    assert summary['performance_report']['runtime']['epochs_completed'] == 1
    assert summary['performance_report']['runtime']['train_samples_per_epoch'] == 8
    assert 'avg_epoch_time_s' in summary['performance_report']['runtime']
    assert 'train_samples_per_sec' in summary['performance_report']['runtime']
    assert 'hotspots' in summary['performance_report']['runtime']
    assert 'train_hotspots' in summary['performance_report']['runtime']
    assert 'eval_hotspots' in summary['performance_report']['runtime']
    assert 'hotspot_diff' in summary['performance_report']['runtime']
    assert summary['performance_report']['runtime']['train_hotspots']['profile_mode'] == 'train'
    assert summary['performance_report']['runtime']['eval_hotspots']['profile_mode'] == 'eval'
    assert summary['performance_report']['runtime']['hotspots']['profile_mode'] == 'eval'
    assert 'top_nodes' in summary['performance_report']['runtime']['hotspots']
    assert 'top_ops' in summary['performance_report']['runtime']['hotspots']
    assert 'top_categories' in summary['performance_report']['runtime']['hotspots']
    assert 'calls' in summary['performance_report']['runtime']['hotspots']['top_ops'][0]
    assert 'avg_ms' in summary['performance_report']['runtime']['hotspots']['top_ops'][0]
    assert 'top_op_deltas' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'top_node_deltas' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'top_category_deltas' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'bottleneck_summary' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'train_total_ms' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'eval_total_ms' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'delta_ms' in summary['performance_report']['runtime']['hotspot_diff']
    assert 'category' in summary['performance_report']['runtime']['hotspot_diff']['top_category_deltas'][0]
    assert 'dominant_train_eval_delta_op' in summary['performance_report']['runtime']['hotspot_diff']['bottleneck_summary']
    assert 'node' in summary['performance_report']['runtime']['hotspot_diff']['top_node_deltas'][0]
    assert 'op' in summary['performance_report']['runtime']['hotspot_diff']['top_op_deltas'][0]
    assert 'train_elapsed_ms' in summary['performance_report']['runtime']['hotspot_diff']['top_op_deltas'][0]
    assert 'eval_elapsed_ms' in summary['performance_report']['runtime']['hotspot_diff']['top_op_deltas'][0]
    assert 'delta_ms' in summary['performance_report']['runtime']['hotspot_diff']['top_op_deltas'][0]
    assert 'state_allocations_per_step' in summary['performance_report']['efficiency']
    assert 'grad_buffer_reuse_ratio' in summary['performance_report']['efficiency']
    assert 'grad_buffer_active_tensor_fraction' in summary['performance_report']['efficiency']
    assert 'grad_buffer_active_byte_fraction' in summary['performance_report']['efficiency']
    assert 'amp_cache_hit_ratio' in summary['performance_report']['efficiency']
    assert 'planner_peak_live_fraction' in summary['performance_report']['efficiency']
    assert 'hints' in summary['performance_report']['bottlenecks']
    assert 'hotspot_bottleneck' in summary['performance_report']['bottlenecks']
    assert 'dominant_train_eval_delta_op' in summary['performance_report']['bottlenecks']['hotspot_bottleneck']
    assert summary['optimizer_runtime']['optimizer_type'] == 'adamw'
    assert summary['optimizer_runtime']['state_tensor_count'] > 0
    assert summary['optimizer_runtime']['state_total_bytes'] > 0
    assert 'scratch_tensor_count' in summary['optimizer_runtime']
    assert 'scratch_total_bytes' in summary['optimizer_runtime']
    assert 'grad_buffer_allocations' in summary['optimizer_runtime']
    assert 'grad_buffer_reset_events' in summary['optimizer_runtime']
    assert 'cache_allocations' in summary['amp_runtime']

    assert row['schema_name'] == 'minicnn.cuda_native.training.metrics.epoch'
    assert row['schema_version'] == 1
    assert row['artifact_kind'] == 'training_metrics_epoch'
    assert row['amp']['enabled'] is True
    assert 'loss_scale' in row['amp']
    assert 'optimizer_runtime' in row
    assert row['optimizer_runtime']['optimizer_type'] == 'adamw'
    assert row['optimizer_runtime']['steps_epoch'] > 0
    assert row['optimizer_runtime']['state_tensor_count'] > 0
    assert 'state_tensor_allocations_epoch' in row['optimizer_runtime']
    assert 'state_tensor_updates_epoch' in row['optimizer_runtime']
    assert 'scratch_tensor_allocations_epoch' in row['optimizer_runtime']
    assert 'scratch_tensor_updates_epoch' in row['optimizer_runtime']
    assert 'grad_buffer_allocations_epoch' in row['optimizer_runtime']
    assert 'grad_buffer_reset_events_epoch' in row['optimizer_runtime']
    assert row['planner']['strategy'] == 'reuse'
    assert 'peak_live_bytes' in row['planner']
    assert 'reuse_events' in row['planner']
    assert 'efficiency' in row
    assert 'grad_buffer_reuse_ratio' in row['efficiency']
    assert 'amp_cache_hit_ratio' in row['efficiency']
    assert 'scratch_allocations_per_step' in row['efficiency']
    assert 'scratch_updates_per_step' in row['efficiency']
