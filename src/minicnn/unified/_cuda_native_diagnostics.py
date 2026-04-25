from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.graph import NativeGraph


def optimizer_runtime_snapshot(optimizer_state: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(optimizer_state.get('optimizer_runtime', {}))
    state_keys = (
        'velocity',
        'adamw_m',
        'adamw_v',
        'adam_m',
        'adam_v',
        'rmsprop_v',
        'rmsprop_buf',
    )
    tensor_count = 0
    total_bytes = 0
    for state_key in state_keys:
        bucket = optimizer_state.get(state_key, {})
        if not isinstance(bucket, dict):
            continue
        for value in bucket.values():
            if isinstance(value, np.ndarray):
                tensor_count += 1
                total_bytes += int(value.nbytes)
    runtime['state_tensor_count'] = tensor_count
    runtime['state_total_bytes'] = total_bytes
    runtime['state_total_kb'] = round(total_bytes / 1024.0, 3)
    scratch_bucket = optimizer_state.get('optimizer_scratch', {})
    scratch_tensor_count = 0
    scratch_total_bytes = 0
    if isinstance(scratch_bucket, dict):
        for value in scratch_bucket.values():
            if isinstance(value, np.ndarray):
                scratch_tensor_count += 1
                scratch_total_bytes += int(value.nbytes)
    runtime['scratch_tensor_count'] = scratch_tensor_count
    runtime['scratch_total_bytes'] = scratch_total_bytes
    runtime['scratch_total_kb'] = round(scratch_total_bytes / 1024.0, 3)
    grad_buffer = optimizer_state.get('grad_buffer', {})
    if isinstance(grad_buffer, dict):
        grad_buffer_tensors = 0
        grad_buffer_bytes = 0
        for value in grad_buffer.values():
            if isinstance(value, np.ndarray):
                grad_buffer_tensors += 1
                grad_buffer_bytes += int(value.nbytes)
        runtime['grad_buffer_tensor_count'] = grad_buffer_tensors
        runtime['grad_buffer_total_bytes'] = grad_buffer_bytes
        runtime['grad_buffer_total_kb'] = round(grad_buffer_bytes / 1024.0, 3)
        active_keys = optimizer_state.get('grad_buffer_active_keys', set())
        if isinstance(active_keys, set):
            active_tensors = 0
            active_bytes = 0
            for key in active_keys:
                value = grad_buffer.get(key)
                if isinstance(value, np.ndarray):
                    active_tensors += 1
                    active_bytes += int(value.nbytes)
            runtime['grad_buffer_active_tensor_count'] = active_tensors
            runtime['grad_buffer_active_total_bytes'] = active_bytes
            runtime['grad_buffer_active_total_kb'] = round(active_bytes / 1024.0, 3)
    return runtime


def profile_hotspots(
    graph: NativeGraph,
    x_sample: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    amp_enabled: bool,
    mode: str,
    top_k: int = 5,
) -> dict[str, Any]:
    from minicnn.cuda_native.debug import TracingForwardExecutor

    if graph.input_spec is None:
        return {}
    tracing_executor = TracingForwardExecutor()
    feeds = {
        graph.input_spec.name: (x_sample.astype(np.float16) if amp_enabled else x_sample),
    }
    profile_params = params
    if amp_enabled:
        profile_params = {
            key: (
                value.astype(np.float16)
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating)
                else value
            )
            for key, value in params.items()
        }
    _ctx, trace = tracing_executor.run(graph, feeds, profile_params, mode=mode)
    trace_summary = trace.summary()
    steps = trace_summary.get('steps', [])
    sorted_steps = sorted(
        (step for step in steps if isinstance(step, dict)),
        key=lambda step: float(step.get('elapsed_ms', 0.0)),
        reverse=True,
    )
    top_nodes = [
        {
            'node': step.get('node'),
            'op': step.get('op'),
            'category': step.get('category'),
            'elapsed_ms': step.get('elapsed_ms'),
        }
        for step in sorted_steps[:top_k]
    ]
    op_totals: dict[str, float] = {}
    op_counts: dict[str, int] = {}
    category_totals: dict[str, float] = {}
    for step in sorted_steps:
        op = str(step.get('op', 'unknown'))
        category = str(step.get('category', 'unknown'))
        elapsed = float(step.get('elapsed_ms', 0.0))
        op_totals[op] = op_totals.get(op, 0.0) + elapsed
        op_counts[op] = op_counts.get(op, 0) + 1
        category_totals[category] = category_totals.get(category, 0.0) + elapsed
    top_ops = [
        {
            'op': op,
            'elapsed_ms': round(elapsed, 3),
            'calls': op_counts[op],
            'avg_ms': round(elapsed / float(max(op_counts[op], 1)), 3),
        }
        for op, elapsed in sorted(op_totals.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]
    top_categories = [
        {'category': category, 'elapsed_ms': round(elapsed, 3)}
        for category, elapsed in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]
    return {
        'profile_mode': mode,
        'sample_batch_size': int(x_sample.shape[0]),
        'trace_total_ms': round(float(trace_summary.get('total_ms', 0.0)), 3),
        'trace_steps': len(steps),
        'top_nodes': top_nodes,
        'top_ops': top_ops,
        'top_categories': top_categories,
    }


def build_hotspot_diff_summary(
    train_hotspots: dict[str, Any],
    eval_hotspots: dict[str, Any],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    train_nodes = {
        (str(item.get('node')), str(item.get('op'))): float(item.get('elapsed_ms', 0.0))
        for item in train_hotspots.get('top_nodes', [])
        if isinstance(item, dict) and item.get('node') is not None
    }
    eval_nodes = {
        (str(item.get('node')), str(item.get('op'))): float(item.get('elapsed_ms', 0.0))
        for item in eval_hotspots.get('top_nodes', [])
        if isinstance(item, dict) and item.get('node') is not None
    }
    train_ops = {
        str(item.get('op')): float(item.get('elapsed_ms', 0.0))
        for item in train_hotspots.get('top_ops', [])
        if isinstance(item, dict) and item.get('op') is not None
    }
    eval_ops = {
        str(item.get('op')): float(item.get('elapsed_ms', 0.0))
        for item in eval_hotspots.get('top_ops', [])
        if isinstance(item, dict) and item.get('op') is not None
    }
    all_ops = sorted(set(train_ops) | set(eval_ops))
    op_deltas = []
    for op in all_ops:
        train_elapsed = train_ops.get(op, 0.0)
        eval_elapsed = eval_ops.get(op, 0.0)
        op_deltas.append(
            {
                'op': op,
                'train_elapsed_ms': round(train_elapsed, 3),
                'eval_elapsed_ms': round(eval_elapsed, 3),
                'delta_ms': round(train_elapsed - eval_elapsed, 3),
                'train_eval_ratio': round(train_elapsed / eval_elapsed, 3) if eval_elapsed > 0.0 else None,
            }
        )
    op_deltas.sort(key=lambda item: abs(float(item.get('delta_ms', 0.0))), reverse=True)
    train_categories = {
        str(item.get('category')): float(item.get('elapsed_ms', 0.0))
        for item in train_hotspots.get('top_categories', [])
        if isinstance(item, dict) and item.get('category') is not None
    }
    eval_categories = {
        str(item.get('category')): float(item.get('elapsed_ms', 0.0))
        for item in eval_hotspots.get('top_categories', [])
        if isinstance(item, dict) and item.get('category') is not None
    }
    all_categories = sorted(set(train_categories) | set(eval_categories))
    category_deltas = []
    for category in all_categories:
        train_elapsed = train_categories.get(category, 0.0)
        eval_elapsed = eval_categories.get(category, 0.0)
        category_deltas.append(
            {
                'category': category,
                'train_elapsed_ms': round(train_elapsed, 3),
                'eval_elapsed_ms': round(eval_elapsed, 3),
                'delta_ms': round(train_elapsed - eval_elapsed, 3),
                'train_eval_ratio': round(train_elapsed / eval_elapsed, 3) if eval_elapsed > 0.0 else None,
            }
        )
    category_deltas.sort(key=lambda item: abs(float(item.get('delta_ms', 0.0))), reverse=True)
    all_nodes = sorted(set(train_nodes) | set(eval_nodes))
    node_deltas = []
    for node_key in all_nodes:
        train_elapsed = train_nodes.get(node_key, 0.0)
        eval_elapsed = eval_nodes.get(node_key, 0.0)
        node_name, op_name = node_key
        node_deltas.append(
            {
                'node': node_name,
                'op': op_name,
                'train_elapsed_ms': round(train_elapsed, 3),
                'eval_elapsed_ms': round(eval_elapsed, 3),
                'delta_ms': round(train_elapsed - eval_elapsed, 3),
                'train_eval_ratio': round(train_elapsed / eval_elapsed, 3) if eval_elapsed > 0.0 else None,
            }
        )
    node_deltas.sort(key=lambda item: abs(float(item.get('delta_ms', 0.0))), reverse=True)
    train_total = float(train_hotspots.get('trace_total_ms', 0.0))
    eval_total = float(eval_hotspots.get('trace_total_ms', 0.0))
    bottleneck_summary = {
        'dominant_train_op': op_deltas[0]['op'] if op_deltas else None,
        'dominant_train_eval_delta_op': op_deltas[0]['op'] if op_deltas else None,
        'dominant_train_eval_delta_node': node_deltas[0]['node'] if node_deltas else None,
        'dominant_train_eval_delta_category': category_deltas[0]['category'] if category_deltas else None,
    }
    return {
        'train_total_ms': round(train_total, 3),
        'eval_total_ms': round(eval_total, 3),
        'delta_ms': round(train_total - eval_total, 3),
        'train_eval_ratio': round(train_total / eval_total, 3) if eval_total > 0.0 else None,
        'bottleneck_summary': bottleneck_summary,
        'top_category_deltas': category_deltas[:top_k],
        'top_node_deltas': node_deltas[:top_k],
        'top_op_deltas': op_deltas[:top_k],
    }
