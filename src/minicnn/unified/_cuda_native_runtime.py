from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.api import assess_cuda_native_support_tier
from minicnn.cuda_native.backward import BackwardExecutor
from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.gpu_training import (
    native_gpu_conv_linear_training_step,
    native_gpu_linear_training_step,
    native_gpu_pool_linear_training_step,
    native_gpu_two_conv_relu_pool_linear_training_step,
    native_gpu_two_linear_relu_training_step,
)
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.planner import make_plan
from minicnn.cuda_native.training import train_step
from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.unified._cuda_native_bridge import (
    _best_checkpoint_path,
    _build_epoch_row,
    _build_training_summary,
    _epoch_log_message,
    _evaluate,
    _init_params,
    _load_numpy_data,
    _make_scheduler,
    _resolve_loss_type,
)


def _optimizer_runtime_snapshot(optimizer_state: dict[str, Any]) -> dict[str, Any]:
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


def _profile_hotspots(
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
        op_totals[op] = op_totals.get(op, 0.0) + float(step.get('elapsed_ms', 0.0))
        op_counts[op] = op_counts.get(op, 0) + 1
        category_totals[category] = category_totals.get(category, 0.0) + float(step.get('elapsed_ms', 0.0))
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


def _build_hotspot_diff_summary(
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
                'train_eval_ratio': (
                    round(train_elapsed / eval_elapsed, 3) if eval_elapsed > 0.0 else None
                ),
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
                'train_eval_ratio': (
                    round(train_elapsed / eval_elapsed, 3) if eval_elapsed > 0.0 else None
                ),
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
                'train_eval_ratio': (
                    round(train_elapsed / eval_elapsed, 3) if eval_elapsed > 0.0 else None
                ),
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


@dataclass
class NativeTrainingContext:
    cfg: dict[str, Any]
    graph: NativeGraph
    params: dict[str, np.ndarray]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    input_shape: tuple[int, ...]
    planner_summary: dict[str, Any]
    batch_size: int
    epochs: int
    lr: float
    optimizer_type: str
    weight_decay: float
    momentum: float
    grad_clip_global: float
    grad_accum_steps: int
    amp: bool
    amp_loss_scale: float
    amp_dynamic_scale: bool
    amp_scale_growth: float
    amp_scale_backoff: float
    amp_scale_window: int
    loss_type: str
    model_cfg: dict[str, Any]
    loss_cfg: dict[str, Any]
    optimizer_cfg: dict[str, Any]
    scheduler_cfg: dict[str, Any]
    support_tier_assessment: dict[str, Any]
    execution_mode: str
    tensor_execution_device: str
    device_runtime: DeviceRuntime


def prepare_training_context(cfg: dict[str, Any], graph: NativeGraph) -> NativeTrainingContext:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    engine_cfg = cfg.get('engine', {})
    optim_cfg = cfg.get('optimizer', {})
    loss_cfg = cfg.get('loss', {})
    scheduler_cfg = cfg.get('scheduler', {})

    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    batch_size = int(train_cfg.get('batch_size', 64))
    epochs = int(train_cfg.get('epochs', 1))
    lr = float(optim_cfg.get('lr', 0.01))
    optimizer_type = str(optim_cfg.get('type', 'SGD')).lower()
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    momentum = float(optim_cfg.get('momentum', 0.0))
    grad_clip_global = float(optim_cfg.get('grad_clip_global', 0.0))
    grad_accum_steps = int(train_cfg.get('grad_accum_steps', 1))
    amp = bool(train_cfg.get('amp', False))
    amp_loss_scale = float(train_cfg.get('amp_loss_scale', 128.0))
    amp_dynamic_scale = bool(train_cfg.get('amp_dynamic_scale', True))
    amp_scale_growth = float(train_cfg.get('amp_scale_growth', 2.0))
    amp_scale_backoff = float(train_cfg.get('amp_scale_backoff', 0.5))
    amp_scale_window = int(train_cfg.get('amp_scale_window', 200))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('seed', 42)))
    loss_type = _resolve_loss_type(loss_cfg)

    x_train, y_train, x_val, y_val = _load_numpy_data(cfg)
    params = _init_params(graph, seed=init_seed)
    planner_strategy = str(engine_cfg.get('planner_strategy', 'reuse'))
    plan = make_plan(graph, strategy=planner_strategy)
    planner_summary_raw = plan.summary()
    buffer_plan = planner_summary_raw.get('buffer_plan', {})
    planner_summary = {
        **planner_summary_raw,
        **({} if not isinstance(buffer_plan, dict) else buffer_plan),
    }
    selected_execution_mode = str(engine_cfg.get('execution_mode', 'reference_numpy') or 'reference_numpy')
    tensor_execution_device = 'gpu' if selected_execution_mode == 'gpu_native' else 'cpu'
    bound_lib = None
    if selected_execution_mode == 'gpu_native':
        from minicnn.core._cuda_library import bind_symbols, ensure_cuda_runtime_available, load_library

        bound_lib = bind_symbols(load_library())
        ensure_cuda_runtime_available(bound_lib)
    device_runtime = DeviceRuntime(
        execution_mode=selected_execution_mode,
        tensor_execution_device=tensor_execution_device,
        bound_lib=bound_lib,
    )
    device_runtime.reserve_from_planner(
        total_bytes=int(planner_summary.get('total_bytes', 0)),
        num_buffers=int(planner_summary.get('num_buffers', 0)),
        workspace_bytes=0,
        buffer_capacities=planner_summary.get('buffers'),
    )

    return NativeTrainingContext(
        cfg=cfg,
        graph=graph,
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        input_shape=input_shape,
        planner_summary=planner_summary,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        momentum=momentum,
        grad_clip_global=grad_clip_global,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        amp_loss_scale=amp_loss_scale,
        amp_dynamic_scale=amp_dynamic_scale,
        amp_scale_growth=amp_scale_growth,
        amp_scale_backoff=amp_scale_backoff,
        amp_scale_window=amp_scale_window,
        loss_type=loss_type,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        optimizer_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        support_tier_assessment=assess_cuda_native_support_tier(cfg),
        execution_mode=selected_execution_mode,
        tensor_execution_device=tensor_execution_device,
        device_runtime=device_runtime,
    )


def _gpu_native_training_plan(graph: NativeGraph) -> dict[str, Any]:
    nodes = list(graph.topological_order())
    ops = [node.op_type for node in nodes]
    if ops == ['Linear']:
        return {'kind': 'linear', 'linear_nodes': [nodes[0]]}
    if ops == ['Flatten', 'Linear']:
        return {'kind': 'linear', 'linear_nodes': [nodes[1]]}
    if ops == ['Linear', 'ReLU', 'Linear']:
        return {'kind': 'two_linear_relu', 'linear_nodes': [nodes[0], nodes[2]], 'relu_node': nodes[1]}
    if ops == ['Flatten', 'Linear', 'ReLU', 'Linear']:
        return {'kind': 'two_linear_relu', 'linear_nodes': [nodes[1], nodes[3]], 'relu_node': nodes[2]}
    if ops == ['MaxPool2d', 'Flatten', 'Linear']:
        return {'kind': 'pool_linear', 'pool_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops in (
        ['Conv2d', 'Flatten', 'Linear'],
        ['Conv2d', 'ReLU', 'Flatten', 'Linear'],
        ['Conv2d', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
    ):
        conv_attrs = dict(getattr(nodes[0], 'attrs', {}) or {})

        def _pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        if bool(conv_attrs.get('bias', False)):
            raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires bias=false.')
        if int(conv_attrs.get('groups', 1)) != 1:
            raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires groups=1.')
        if _pair(conv_attrs.get('stride', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires stride=1.')
        if _pair(conv_attrs.get('padding', 0), 0) != (0, 0):
            raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires padding=0.')
        if _pair(conv_attrs.get('dilation', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires dilation=1.')
        if ops == ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear']:
            conv2_attrs = dict(getattr(nodes[2], 'attrs', {}) or {})
            if bool(conv2_attrs.get('bias', False)):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires bias=false.')
            if int(conv2_attrs.get('groups', 1)) != 1:
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires groups=1.')
            if _pair(conv2_attrs.get('stride', 1), 1) != (1, 1):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires stride=1.')
            if _pair(conv2_attrs.get('padding', 0), 0) != (0, 0):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires padding=0.')
            if _pair(conv2_attrs.get('dilation', 1), 1) != (1, 1):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires dilation=1.')
            return {
                'kind': 'two_conv_relu_pool_linear',
                'conv_nodes': [nodes[0], nodes[2]],
                'linear_nodes': [nodes[-1]],
            }
        has_relu = 'ReLU' in ops
        has_pool = 'MaxPool2d' in ops
        linear_node = nodes[-1]
        return {
            'kind': 'conv_linear',
            'conv_node': nodes[0],
            'linear_nodes': [linear_node],
            'apply_relu_activation': has_relu,
            'apply_maxpool': has_pool,
        }
    raise ValueError(
        'cuda_native gpu_native train-native currently supports only '
        'ops=[Linear], ops=[Flatten, Linear], ops=[Linear, ReLU, Linear], '
        'ops=[Flatten, Linear, ReLU, Linear], ops=[MaxPool2d, Flatten, Linear], '
        'ops=[Conv2d, Flatten, Linear], ops=[Conv2d, ReLU, Flatten, Linear], '
        'ops=[Conv2d, MaxPool2d, Flatten, Linear], or '
        'ops=[Conv2d, ReLU, MaxPool2d, Flatten, Linear], or '
        'ops=[Conv2d, ReLU, Conv2d, ReLU, MaxPool2d, Flatten, Linear], '
        f'got {ops}.'
    )


def _validate_gpu_native_training_context(ctx: NativeTrainingContext) -> None:
    plan = _gpu_native_training_plan(ctx.graph)
    if ctx.loss_type != 'cross_entropy' and plan['kind'] != 'linear':
        raise ValueError('cuda_native gpu_native train-native currently supports MSELoss/BCEWithLogitsLoss only for the Linear subset.')
    if plan['kind'] == 'linear':
        if ctx.optimizer_type not in {'sgd', 'adam', 'adamw', 'rmsprop'}:
            raise ValueError('cuda_native gpu_native Linear train-native currently supports optimizer.type in {SGD, Adam, AdamW, RMSprop}.')
        if ctx.optimizer_type == 'adam' and ctx.weight_decay != 0.0:
            raise ValueError('cuda_native gpu_native Linear train-native currently requires Adam weight_decay=0.0; use AdamW for decoupled weight decay.')
    else:
        if ctx.optimizer_type != 'sgd':
            raise ValueError('cuda_native gpu_native non-Linear train-native currently supports only optimizer.type=SGD.')
        if ctx.weight_decay != 0.0:
            raise ValueError('cuda_native gpu_native non-Linear train-native currently requires optimizer.weight_decay=0.0.')
    if ctx.grad_accum_steps != 1:
        raise ValueError('cuda_native gpu_native train-native currently requires train.grad_accum_steps=1.')
    if ctx.amp:
        raise ValueError('cuda_native gpu_native train-native currently requires train.amp=false.')


def _merge_gpu_native_step_runtime(ctx: NativeTrainingContext, step_summary: dict[str, Any]) -> None:
    for attr in (
        'host_to_device_transfer_events',
        'host_to_device_transfer_bytes',
        'device_to_host_transfer_events',
        'device_to_host_transfer_bytes',
        'allocation_events',
        'allocated_bytes',
        'synchronization_events',
        'device_pointer_allocation_events',
        'device_pointer_free_events',
        'device_pointer_bytes',
        'device_sync_to_host_events',
        'device_sync_to_device_events',
    ):
        setattr(ctx.device_runtime, attr, int(getattr(ctx.device_runtime, attr)) + int(step_summary.get(attr, 0)))
    ctx.device_runtime.device_pointer_live_bytes += int(step_summary.get('device_pointer_live_bytes', 0))
    for kind, count in dict(step_summary.get('execution_kinds', {})).items():
        for _ in range(int(count)):
            ctx.device_runtime.record_execution(str(kind), node_count=0)


def run_training_loop(
    ctx: NativeTrainingContext,
    *,
    run_dir: Path,
) -> tuple[dict[str, np.ndarray], float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics_path = run_dir / 'metrics.jsonl'
    fwd = ForwardExecutor()
    bwd = BackwardExecutor()
    optimizer_view = SimpleNamespace(lr=ctx.lr)
    scheduler, scheduler_kind = _make_scheduler(ctx.scheduler_cfg, optimizer_view)
    optimizer_state: dict[str, Any] = {}
    gpu_training_plan = None
    if ctx.execution_mode == 'gpu_native':
        _validate_gpu_native_training_context(ctx)
        gpu_training_plan = _gpu_native_training_plan(ctx.graph)
        optimizer_state['optimizer_runtime'] = {
            'optimizer_type': ctx.optimizer_type,
            'steps': 0,
        }
        optimizer_state['velocity'] = {}
        optimizer_state['adam_m'] = {}
        optimizer_state['adam_v'] = {}
        optimizer_state['adamw_m'] = {}
        optimizer_state['adamw_v'] = {}
        optimizer_state['rmsprop_v'] = {}
        optimizer_state['rmsprop_buf'] = {}
    best_val_acc = float('-inf')
    best_params = dict(ctx.params)
    params = ctx.params
    rng = np.random.default_rng(int(ctx.cfg.get('train', {}).get('init_seed', ctx.cfg.get('dataset', {}).get('seed', 42))))
    prev_amp_snapshot = {
        'skipped_steps': 0,
        'overflow_steps': 0,
        'finite_steps': 0,
        'cache_allocations': 0,
        'cache_updates': 0,
        'cache_hits': 0,
    }
    prev_optimizer_snapshot = {
        'steps': 0,
        'state_tensor_allocations': 0,
        'state_tensor_updates': 0,
        'scratch_tensor_allocations': 0,
        'scratch_tensor_updates': 0,
        'grad_buffer_allocations': 0,
        'grad_buffer_reuses': 0,
        'grad_buffer_reset_events': 0,
        'grad_buffer_zeroed_tensors': 0,
    }
    epoch_times: list[float] = []
    planner_epoch_state = {
        'strategy': str(ctx.planner_summary.get('strategy', 'unknown')),
        'num_buffers': int(ctx.planner_summary.get('num_buffers', 0)),
        'total_bytes': int(ctx.planner_summary.get('total_bytes', 0)),
        'total_kb': float(ctx.planner_summary.get('total_kb', 0.0)),
        'peak_live_bytes': int(ctx.planner_summary.get('peak_live_bytes', 0)),
        'peak_live_kb': float(ctx.planner_summary.get('peak_live_kb', 0.0)),
        'reuse_events': int(ctx.planner_summary.get('reuse_events', 0)),
        'reuse_slack_bytes': int(ctx.planner_summary.get('reuse_slack_bytes', 0)),
    }

    with metrics_path.open('w', encoding='utf-8') as mf:
        for epoch in range(1, ctx.epochs + 1):
            t0 = time.perf_counter()
            idx = rng.permutation(ctx.x_train.shape[0])
            x_shuf, y_shuf = ctx.x_train[idx], ctx.y_train[idx]
            running_loss = 0.0
            seen = 0
            num_batches = (x_shuf.shape[0] + ctx.batch_size - 1) // ctx.batch_size
            batch_idx = 0
            for i in range(0, x_shuf.shape[0], ctx.batch_size):
                batch_idx += 1
                xb = x_shuf[i:i + ctx.batch_size]
                yb = y_shuf[i:i + ctx.batch_size]
                if xb.shape[0] == 0:
                    continue
                apply_optimizer_step = (
                    batch_idx % ctx.grad_accum_steps == 0
                    or batch_idx == num_batches
                )
                if ctx.execution_mode == 'gpu_native':
                    if not apply_optimizer_step:
                        raise ValueError('cuda_native gpu_native train-native currently requires train.grad_accum_steps=1.')
                    assert gpu_training_plan is not None
                    velocity_state = optimizer_state.setdefault('velocity', {})
                    flat_xb = xb.reshape(xb.shape[0], -1)
                    if gpu_training_plan['kind'] == 'linear':
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        weight_key = f'_w_{gpu_linear_node.name}'
                        bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_linear_training_step(
                            flat_xb,
                            yb,
                            params[weight_key],
                            params[bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            loss_type=ctx.loss_type,
                            optimizer_type=ctx.optimizer_type,
                            weight_decay=float(ctx.weight_decay),
                            beta1=float(ctx.optimizer_cfg.get('beta1', 0.9)),
                            beta2=float(ctx.optimizer_cfg.get('beta2', 0.999)),
                            eps=float(ctx.optimizer_cfg.get('eps', 1e-8)),
                            rmsprop_alpha=float(ctx.optimizer_cfg.get('alpha', 0.99)),
                            step_index=int(optimizer_state.get('optimizer_runtime', {}).get('steps', 0)) + 1,
                            weight_velocity=velocity_state.get(weight_key),
                            bias_velocity=velocity_state.get(bias_key),
                            weight_m=(
                                optimizer_state.setdefault('adamw_m' if ctx.optimizer_type == 'adamw' else 'adam_m', {}).get(weight_key)
                                if ctx.optimizer_type in {'adam', 'adamw'} else None
                            ),
                            weight_v=(
                                optimizer_state.setdefault('adamw_v' if ctx.optimizer_type == 'adamw' else 'adam_v', {}).get(weight_key)
                                if ctx.optimizer_type in {'adam', 'adamw'} else None
                            ),
                            bias_m=(
                                optimizer_state.setdefault('adamw_m' if ctx.optimizer_type == 'adamw' else 'adam_m', {}).get(bias_key)
                                if ctx.optimizer_type in {'adam', 'adamw'} else None
                            ),
                            bias_v=(
                                optimizer_state.setdefault('adamw_v' if ctx.optimizer_type == 'adamw' else 'adam_v', {}).get(bias_key)
                                if ctx.optimizer_type in {'adam', 'adamw'} else None
                            ),
                            weight_rmsprop_v=(
                                optimizer_state.setdefault('rmsprop_v', {}).get(weight_key)
                                if ctx.optimizer_type == 'rmsprop' else None
                            ),
                            weight_rmsprop_buf=(
                                optimizer_state.setdefault('rmsprop_buf', {}).get(weight_key)
                                if ctx.optimizer_type == 'rmsprop' else None
                            ),
                            bias_rmsprop_v=(
                                optimizer_state.setdefault('rmsprop_v', {}).get(bias_key)
                                if ctx.optimizer_type == 'rmsprop' else None
                            ),
                            bias_rmsprop_buf=(
                                optimizer_state.setdefault('rmsprop_buf', {}).get(bias_key)
                                if ctx.optimizer_type == 'rmsprop' else None
                            ),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[weight_key] = step.updated_weight
                        params[bias_key] = step.updated_bias
                        velocity_state[weight_key] = step.updated_weight_velocity
                        velocity_state[bias_key] = step.updated_bias_velocity
                        if ctx.optimizer_type in {'adam', 'adamw'}:
                            m_bucket = optimizer_state.setdefault('adamw_m' if ctx.optimizer_type == 'adamw' else 'adam_m', {})
                            v_bucket = optimizer_state.setdefault('adamw_v' if ctx.optimizer_type == 'adamw' else 'adam_v', {})
                            m_bucket[weight_key] = step.updated_weight_m
                            v_bucket[weight_key] = step.updated_weight_v
                            m_bucket[bias_key] = step.updated_bias_m
                            v_bucket[bias_key] = step.updated_bias_v
                        if ctx.optimizer_type == 'rmsprop':
                            v_bucket = optimizer_state.setdefault('rmsprop_v', {})
                            buf_bucket = optimizer_state.setdefault('rmsprop_buf', {})
                            v_bucket[weight_key] = step.updated_weight_rmsprop_v
                            buf_bucket[weight_key] = step.updated_weight_rmsprop_buf
                            v_bucket[bias_key] = step.updated_bias_rmsprop_v
                            buf_bucket[bias_key] = step.updated_bias_rmsprop_buf
                    elif gpu_training_plan['kind'] == 'two_linear_relu':
                        first_linear, second_linear = gpu_training_plan['linear_nodes']
                        w1_key = f'_w_{first_linear.name}'
                        b1_key = f'_b_{first_linear.name}'
                        w2_key = f'_w_{second_linear.name}'
                        b2_key = f'_b_{second_linear.name}'
                        step = native_gpu_two_linear_relu_training_step(
                            flat_xb,
                            yb,
                            params[w1_key],
                            params[b1_key],
                            params[w2_key],
                            params[b2_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            weight1_velocity=velocity_state.get(w1_key),
                            bias1_velocity=velocity_state.get(b1_key),
                            weight2_velocity=velocity_state.get(w2_key),
                            bias2_velocity=velocity_state.get(b2_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[w1_key] = step.updated_weight1
                        params[b1_key] = step.updated_bias1
                        params[w2_key] = step.updated_weight2
                        params[b2_key] = step.updated_bias2
                        velocity_state[w1_key] = step.updated_weight1_velocity
                        velocity_state[b1_key] = step.updated_bias1_velocity
                        velocity_state[w2_key] = step.updated_weight2_velocity
                        velocity_state[b2_key] = step.updated_bias2_velocity
                    elif gpu_training_plan['kind'] == 'pool_linear':
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        weight_key = f'_w_{gpu_linear_node.name}'
                        bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_pool_linear_training_step(
                            xb,
                            yb,
                            params[weight_key],
                            params[bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            weight_velocity=velocity_state.get(weight_key),
                            bias_velocity=velocity_state.get(bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[weight_key] = step.updated_weight
                        params[bias_key] = step.updated_bias
                        velocity_state[weight_key] = step.updated_weight_velocity
                        velocity_state[bias_key] = step.updated_bias_velocity
                    elif gpu_training_plan['kind'] == 'conv_linear':
                        conv_node = gpu_training_plan['conv_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        conv_weight_key = f'_w_{conv_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_conv_linear_training_step(
                            xb,
                            yb,
                            params[conv_weight_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            conv_weight_velocity=velocity_state.get(conv_weight_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            apply_relu_activation=bool(gpu_training_plan.get('apply_relu_activation', False)),
                            apply_maxpool=bool(gpu_training_plan.get('apply_maxpool', False)),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[conv_weight_key] = step.updated_conv_weight
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[conv_weight_key] = step.updated_conv_weight_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    else:
                        conv1_node, conv2_node = gpu_training_plan['conv_nodes']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        conv1_weight_key = f'_w_{conv1_node.name}'
                        conv2_weight_key = f'_w_{conv2_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_two_conv_relu_pool_linear_training_step(
                            xb,
                            yb,
                            params[conv1_weight_key],
                            params[conv2_weight_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            conv1_weight_velocity=velocity_state.get(conv1_weight_key),
                            conv2_weight_velocity=velocity_state.get(conv2_weight_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[conv1_weight_key] = step.updated_conv1_weight
                        params[conv2_weight_key] = step.updated_conv2_weight
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[conv1_weight_key] = step.updated_conv1_weight_velocity
                        velocity_state[conv2_weight_key] = step.updated_conv2_weight_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    loss_val = float(step.loss_mean)
                    _merge_gpu_native_step_runtime(ctx, step.runtime_summary)
                    optimizer_runtime = optimizer_state.setdefault('optimizer_runtime', {})
                    optimizer_runtime['optimizer_type'] = ctx.optimizer_type
                    optimizer_runtime['steps'] = int(optimizer_runtime.get('steps', 0)) + 1
                    ctx.device_runtime.record_execution(
                        'gpu_native_train_batch',
                        input_name=ctx.graph.input_spec.name if ctx.graph.input_spec is not None else 'input',
                        output_name=ctx.graph.output_spec.name if ctx.graph.output_spec is not None else 'output',
                        node_count=len(ctx.graph.nodes),
                    )
                    ctx.device_runtime.synchronize('gpu-native-train-batch')
                else:
                    xb_device = ctx.device_runtime.stage_to_device(
                        xb,
                        name=ctx.graph.input_spec.name if ctx.graph.input_spec else 'input',
                        prefer_reserved=True,
                    )
                    loss_val, params = train_step(
                        ctx.graph,
                        xb_device.data,
                        yb,
                        params,
                        lr=optimizer_view.lr,
                        loss_type=ctx.loss_type,
                        optimizer_type=ctx.optimizer_type,
                        weight_decay=ctx.weight_decay,
                        momentum=ctx.momentum,
                        label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                        grad_accum_steps=ctx.grad_accum_steps,
                        apply_optimizer_step=apply_optimizer_step,
                        amp_enabled=ctx.amp,
                        amp_loss_scale=ctx.amp_loss_scale,
                        amp_dynamic_scale=ctx.amp_dynamic_scale,
                        amp_scale_growth=ctx.amp_scale_growth,
                        amp_scale_backoff=ctx.amp_scale_backoff,
                        amp_scale_window=ctx.amp_scale_window,
                        optimizer_state=optimizer_state,
                        grad_clip_global=ctx.grad_clip_global,
                        fwd_executor=fwd,
                        bwd_executor=bwd,
                    )
                    ctx.device_runtime.record_execution(
                        'train_batch',
                        input_name=ctx.graph.input_spec.name if ctx.graph.input_spec is not None else 'input',
                        output_name=ctx.graph.output_spec.name if ctx.graph.output_spec is not None else 'output',
                        node_count=len(ctx.graph.nodes),
                    )
                    ctx.device_runtime.synchronize('train-batch')
                    ctx.device_runtime.release_buffer(xb_device)
                running_loss += loss_val * xb.shape[0]
                seen += xb.shape[0]
            train_loss = running_loss / max(seen, 1)
            val_metrics = _evaluate(
                ctx.graph,
                ctx.x_val,
                ctx.y_val,
                params,
                ctx.batch_size,
                ctx.loss_type,
                amp_enabled=ctx.amp,
                device_runtime=ctx.device_runtime,
            )
            if scheduler is not None:
                if scheduler_kind == 'plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            epoch_time = time.perf_counter() - t0
            epoch_times.append(epoch_time)
            amp_epoch_state: dict[str, Any] | None = None
            optimizer_epoch_state: dict[str, Any] | None = None
            if ctx.amp:
                amp_runtime = dict(optimizer_state.get('amp', {}))
                amp_epoch_state = {
                    'enabled': True,
                    'loss_scale': float(amp_runtime.get('loss_scale', ctx.amp_loss_scale)),
                    'skipped_steps_epoch': int(amp_runtime.get('skipped_steps', 0)) - int(prev_amp_snapshot['skipped_steps']),
                    'overflow_steps_epoch': int(amp_runtime.get('overflow_steps', 0)) - int(prev_amp_snapshot['overflow_steps']),
                    'finite_steps_epoch': int(amp_runtime.get('finite_steps', 0)) - int(prev_amp_snapshot['finite_steps']),
                    'cache_allocations_epoch': int(amp_runtime.get('cache_allocations', 0)) - int(prev_amp_snapshot['cache_allocations']),
                    'cache_updates_epoch': int(amp_runtime.get('cache_updates', 0)) - int(prev_amp_snapshot['cache_updates']),
                    'cache_hits_epoch': int(amp_runtime.get('cache_hits', 0)) - int(prev_amp_snapshot['cache_hits']),
                }
                prev_amp_snapshot = {
                    'skipped_steps': int(amp_runtime.get('skipped_steps', 0)),
                    'overflow_steps': int(amp_runtime.get('overflow_steps', 0)),
                    'finite_steps': int(amp_runtime.get('finite_steps', 0)),
                    'cache_allocations': int(amp_runtime.get('cache_allocations', 0)),
                    'cache_updates': int(amp_runtime.get('cache_updates', 0)),
                    'cache_hits': int(amp_runtime.get('cache_hits', 0)),
                }
            optimizer_runtime = _optimizer_runtime_snapshot(optimizer_state)
            optimizer_epoch_state = {
                'optimizer_type': str(optimizer_runtime.get('optimizer_type', ctx.optimizer_type)),
                'steps': int(optimizer_runtime.get('steps', 0)),
                'state_tensor_count': int(optimizer_runtime.get('state_tensor_count', 0)),
                'state_total_bytes': int(optimizer_runtime.get('state_total_bytes', 0)),
                'state_total_kb': float(optimizer_runtime.get('state_total_kb', 0.0)),
                'scratch_tensor_count': int(optimizer_runtime.get('scratch_tensor_count', 0)),
                'scratch_total_bytes': int(optimizer_runtime.get('scratch_total_bytes', 0)),
                'scratch_total_kb': float(optimizer_runtime.get('scratch_total_kb', 0.0)),
                'grad_buffer_tensor_count': int(optimizer_runtime.get('grad_buffer_tensor_count', 0)),
                'grad_buffer_total_bytes': int(optimizer_runtime.get('grad_buffer_total_bytes', 0)),
                'grad_buffer_total_kb': float(optimizer_runtime.get('grad_buffer_total_kb', 0.0)),
                'grad_buffer_active_tensor_count': int(optimizer_runtime.get('grad_buffer_active_tensor_count', 0)),
                'grad_buffer_active_total_bytes': int(optimizer_runtime.get('grad_buffer_active_total_bytes', 0)),
                'grad_buffer_active_total_kb': float(optimizer_runtime.get('grad_buffer_active_total_kb', 0.0)),
                'state_tensor_allocations_epoch': (
                    int(optimizer_runtime.get('state_tensor_allocations', 0))
                    - int(prev_optimizer_snapshot['state_tensor_allocations'])
                ),
                'state_tensor_updates_epoch': (
                    int(optimizer_runtime.get('state_tensor_updates', 0))
                    - int(prev_optimizer_snapshot['state_tensor_updates'])
                ),
                'scratch_tensor_allocations_epoch': (
                    int(optimizer_runtime.get('scratch_tensor_allocations', 0))
                    - int(prev_optimizer_snapshot['scratch_tensor_allocations'])
                ),
                'scratch_tensor_updates_epoch': (
                    int(optimizer_runtime.get('scratch_tensor_updates', 0))
                    - int(prev_optimizer_snapshot['scratch_tensor_updates'])
                ),
                'grad_buffer_allocations_epoch': (
                    int(optimizer_runtime.get('grad_buffer_allocations', 0))
                    - int(prev_optimizer_snapshot['grad_buffer_allocations'])
                ),
                'grad_buffer_reuses_epoch': (
                    int(optimizer_runtime.get('grad_buffer_reuses', 0))
                    - int(prev_optimizer_snapshot['grad_buffer_reuses'])
                ),
                'grad_buffer_reset_events_epoch': (
                    int(optimizer_runtime.get('grad_buffer_reset_events', 0))
                    - int(prev_optimizer_snapshot['grad_buffer_reset_events'])
                ),
                'grad_buffer_zeroed_tensors_epoch': (
                    int(optimizer_runtime.get('grad_buffer_zeroed_tensors', 0))
                    - int(prev_optimizer_snapshot['grad_buffer_zeroed_tensors'])
                ),
                'steps_epoch': int(optimizer_runtime.get('steps', 0)) - int(prev_optimizer_snapshot['steps']),
            }
            prev_optimizer_snapshot = {
                'steps': int(optimizer_runtime.get('steps', 0)),
                'state_tensor_allocations': int(optimizer_runtime.get('state_tensor_allocations', 0)),
                'state_tensor_updates': int(optimizer_runtime.get('state_tensor_updates', 0)),
                'scratch_tensor_allocations': int(optimizer_runtime.get('scratch_tensor_allocations', 0)),
                'scratch_tensor_updates': int(optimizer_runtime.get('scratch_tensor_updates', 0)),
                'grad_buffer_allocations': int(optimizer_runtime.get('grad_buffer_allocations', 0)),
                'grad_buffer_reuses': int(optimizer_runtime.get('grad_buffer_reuses', 0)),
                'grad_buffer_reset_events': int(optimizer_runtime.get('grad_buffer_reset_events', 0)),
                'grad_buffer_zeroed_tensors': int(optimizer_runtime.get('grad_buffer_zeroed_tensors', 0)),
            }
            row = _build_epoch_row(
                epoch=epoch,
                train_loss=train_loss,
                val_metrics=val_metrics,
                lr=float(optimizer_view.lr),
                epoch_time_s=epoch_time,
                amp_state=amp_epoch_state,
                optimizer_state=optimizer_epoch_state,
                planner_state=planner_epoch_state,
                device_runtime_state=ctx.device_runtime.summary(),
                support_tier_assessment=ctx.support_tier_assessment,
                execution_mode=ctx.execution_mode,
                tensor_execution_device=ctx.tensor_execution_device,
            )
            mf.write(json.dumps(row) + '\n')
            mf.flush()
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_params = {k: v.copy() for k, v in params.items()}
            print(
                _epoch_log_message(
                    epoch=epoch,
                    epochs=ctx.epochs,
                    train_loss=train_loss,
                    val_acc=val_metrics['acc'],
                    lr=float(optimizer_view.lr),
                    epoch_time_s=epoch_time,
                    amp_state=amp_epoch_state,
                    optimizer_state=optimizer_epoch_state,
                )
            )

    amp_runtime = dict(optimizer_state.get('amp', {}))
    optimizer_runtime = _optimizer_runtime_snapshot(optimizer_state)
    total_epoch_time = float(sum(epoch_times))
    epochs_completed = len(epoch_times)
    train_samples_per_epoch = int(ctx.x_train.shape[0])
    train_hotspot_profile = {}
    eval_hotspot_profile = {}
    if ctx.x_val.shape[0] > 0:
        eval_sample_batch = ctx.x_val[: min(ctx.batch_size, ctx.x_val.shape[0])]
        eval_hotspot_profile = _profile_hotspots(
            ctx.graph,
            eval_sample_batch,
            best_params,
            amp_enabled=ctx.amp,
            mode='eval',
        )
    if ctx.x_train.shape[0] > 0:
        train_sample_batch = ctx.x_train[: min(ctx.batch_size, ctx.x_train.shape[0])]
        train_hotspot_profile = _profile_hotspots(
            ctx.graph,
            train_sample_batch,
            best_params,
            amp_enabled=ctx.amp,
            mode='train',
        )
    runtime_profile = {
        'device_runtime': dict(ctx.device_runtime.summary()),
        'epochs_completed': epochs_completed,
        'train_samples_per_epoch': train_samples_per_epoch,
        'val_samples_per_epoch': int(ctx.x_val.shape[0]),
        'total_epoch_time_s': round(total_epoch_time, 6),
        'avg_epoch_time_s': round(total_epoch_time / float(max(epochs_completed, 1)), 6),
        'min_epoch_time_s': round(min(epoch_times), 6) if epoch_times else 0.0,
        'max_epoch_time_s': round(max(epoch_times), 6) if epoch_times else 0.0,
        'train_samples_per_sec': (
            round((train_samples_per_epoch * epochs_completed) / total_epoch_time, 6)
            if total_epoch_time > 0.0 and epochs_completed > 0
            else 0.0
        ),
        'train_hotspots': train_hotspot_profile,
        'eval_hotspots': eval_hotspot_profile,
        'hotspots': eval_hotspot_profile,
        'hotspot_diff': _build_hotspot_diff_summary(train_hotspot_profile, eval_hotspot_profile),
    }
    return best_params, best_val_acc, amp_runtime, optimizer_runtime, runtime_profile


def finalize_training_run(
    ctx: NativeTrainingContext,
    *,
    run_dir: Path,
    best_params: dict[str, np.ndarray],
    best_val_acc: float,
    amp_runtime: dict[str, Any],
    optimizer_runtime: dict[str, Any],
    runtime_profile: dict[str, Any],
    capabilities: dict[str, Any],
) -> Path:
    best_path = _best_checkpoint_path(run_dir)
    np.savez(str(best_path), **best_params)
    summary = _build_training_summary(
        run_dir=run_dir,
        best_path=best_path,
        best_val_acc=best_val_acc,
        input_shape=ctx.input_shape,
        model_cfg=ctx.model_cfg,
        loss_cfg=ctx.loss_cfg,
        optimizer_cfg=ctx.optimizer_cfg,
        train_cfg=ctx.cfg.get('train', {}),
        amp_runtime=amp_runtime,
        optimizer_runtime=optimizer_runtime,
        planner_summary=ctx.planner_summary,
        runtime_profile=runtime_profile,
        scheduler_cfg=ctx.scheduler_cfg,
        epochs=ctx.epochs,
        capabilities=capabilities,
        support_tier_assessment=ctx.support_tier_assessment,
        execution_mode=ctx.execution_mode,
        tensor_execution_device=ctx.tensor_execution_device,
        device_runtime_state=ctx.device_runtime.summary(),
    )
    dump_summary(run_dir, summary)
    return run_dir


def train_and_summarize_native_backend(
    cfg: dict[str, Any],
    *,
    graph: NativeGraph,
    capabilities: dict[str, Any],
) -> Path:
    ctx = prepare_training_context(cfg, graph)
    run_dir = create_run_dir(cfg)
    best_params, best_val_acc, amp_runtime, optimizer_runtime, runtime_profile = run_training_loop(ctx, run_dir=run_dir)
    return finalize_training_run(
        ctx,
        run_dir=run_dir,
        best_params=best_params,
        best_val_acc=best_val_acc,
        amp_runtime=amp_runtime,
        optimizer_runtime=optimizer_runtime,
        runtime_profile=runtime_profile,
        capabilities=capabilities,
    )
