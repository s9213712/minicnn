from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.backward import BackwardExecutor
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.gpu_training import (
    native_gpu_avgpool_linear_training_step,
    native_gpu_batchnorm_linear_training_step,
    native_gpu_conv_linear_training_step,
    native_gpu_depthwise_layernorm2d_linear_training_step,
    native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step,
    native_gpu_depthwise_layernorm2d_pointwise_linear_training_step,
    native_gpu_global_avgpool_linear_training_step,
    native_gpu_groupnorm_linear_training_step,
    native_gpu_layernorm2d_linear_training_step,
    native_gpu_linear_training_step,
    native_gpu_pool_linear_training_step,
    native_gpu_two_conv_relu_pool_linear_training_step,
    native_gpu_two_linear_relu_training_step,
)
from minicnn.cuda_native.training import train_step
from minicnn.unified._cuda_native_bridge import (
    _build_epoch_row,
    _epoch_log_message,
    _evaluate,
    _make_scheduler,
)
from minicnn.unified._cuda_native_context import NativeTrainingContext
from minicnn.unified._cuda_native_training_plan import (
    _gpu_native_training_plan,
    _merge_gpu_native_step_runtime,
    _validate_gpu_native_training_context,
)
from minicnn.unified._cuda_native_diagnostics import (
    build_hotspot_diff_summary,
    optimizer_runtime_snapshot,
    profile_hotspots,
)

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
            running_correct = 0
            correct_seen = 0
            seen = 0
            num_batches = (x_shuf.shape[0] + ctx.batch_size - 1) // ctx.batch_size
            batch_idx = 0
            gpu_accum_x: list[np.ndarray] = []
            gpu_accum_y: list[np.ndarray] = []
            for i in range(0, x_shuf.shape[0], ctx.batch_size):
                batch_idx += 1
                xb = x_shuf[i:i + ctx.batch_size]
                yb = y_shuf[i:i + ctx.batch_size]
                if xb.shape[0] == 0:
                    continue
                batch_correct: int | None = None
                apply_optimizer_step = (
                    batch_idx % ctx.grad_accum_steps == 0
                    or batch_idx == num_batches
                )
                if ctx.execution_mode == 'gpu_native':
                    gpu_accum_x.append(xb)
                    gpu_accum_y.append(yb)
                    if not apply_optimizer_step:
                        continue
                    if len(gpu_accum_x) > 1:
                        xb = np.concatenate(gpu_accum_x, axis=0)
                        yb = np.concatenate(gpu_accum_y, axis=0)
                    gpu_accum_x.clear()
                    gpu_accum_y.clear()
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
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            optimizer_type=ctx.optimizer_type,
                            weight_decay=float(ctx.weight_decay),
                            grad_clip_value=float(ctx.grad_clip_global),
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
                    elif gpu_training_plan['kind'] == 'two_linear_activation':
                        first_linear, second_linear = gpu_training_plan['linear_nodes']
                        activation_node = gpu_training_plan['activation_node']
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
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            weight1_velocity=velocity_state.get(w1_key),
                            bias1_velocity=velocity_state.get(b1_key),
                            weight2_velocity=velocity_state.get(w2_key),
                            bias2_velocity=velocity_state.get(b2_key),
                            activation=str(activation_node.op_type),
                            activation_alpha=float(activation_node.attrs.get('negative_slope', 0.01)),
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
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            weight_velocity=velocity_state.get(weight_key),
                            bias_velocity=velocity_state.get(bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[weight_key] = step.updated_weight
                        params[bias_key] = step.updated_bias
                        velocity_state[weight_key] = step.updated_weight_velocity
                        velocity_state[bias_key] = step.updated_bias_velocity
                    elif gpu_training_plan['kind'] == 'global_avgpool_linear':
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        weight_key = f'_w_{gpu_linear_node.name}'
                        bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_global_avgpool_linear_training_step(
                            xb,
                            yb,
                            params[weight_key],
                            params[bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            weight_velocity=velocity_state.get(weight_key),
                            bias_velocity=velocity_state.get(bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[weight_key] = step.updated_weight
                        params[bias_key] = step.updated_bias
                        velocity_state[weight_key] = step.updated_weight_velocity
                        velocity_state[bias_key] = step.updated_bias_velocity
                    elif gpu_training_plan['kind'] == 'avgpool_linear':
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        weight_key = f'_w_{gpu_linear_node.name}'
                        bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_avgpool_linear_training_step(
                            xb,
                            yb,
                            params[weight_key],
                            params[bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            weight_velocity=velocity_state.get(weight_key),
                            bias_velocity=velocity_state.get(bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[weight_key] = step.updated_weight
                        params[bias_key] = step.updated_bias
                        velocity_state[weight_key] = step.updated_weight_velocity
                        velocity_state[bias_key] = step.updated_bias_velocity
                    elif gpu_training_plan['kind'] == 'batchnorm_linear':
                        bn_node = gpu_training_plan['batchnorm_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        bn_weight_key = f'_w_{bn_node.name}'
                        bn_bias_key = f'_b_{bn_node.name}'
                        running_mean_key = f'_running_mean_{bn_node.name}'
                        running_var_key = f'_running_var_{bn_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_batchnorm_linear_training_step(
                            xb,
                            yb,
                            params[bn_weight_key],
                            params[bn_bias_key],
                            params[running_mean_key],
                            params[running_var_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            bn_eps=float(bn_node.attrs.get('eps', 1e-5)),
                            bn_momentum=float(bn_node.attrs.get('momentum', 0.1)),
                            bn_weight_velocity=velocity_state.get(bn_weight_key),
                            bn_bias_velocity=velocity_state.get(bn_bias_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[bn_weight_key] = step.updated_bn_weight
                        params[bn_bias_key] = step.updated_bn_bias
                        params[running_mean_key] = step.updated_running_mean
                        params[running_var_key] = step.updated_running_var
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[bn_weight_key] = step.updated_bn_weight_velocity
                        velocity_state[bn_bias_key] = step.updated_bn_bias_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    elif gpu_training_plan['kind'] == 'layernorm2d_linear':
                        norm_node = gpu_training_plan['layernorm2d_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        norm_weight_key = f'_w_{norm_node.name}'
                        norm_bias_key = f'_b_{norm_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_layernorm2d_linear_training_step(
                            xb,
                            yb,
                            params[norm_weight_key],
                            params[norm_bias_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
                            norm_weight_velocity=velocity_state.get(norm_weight_key),
                            norm_bias_velocity=velocity_state.get(norm_bias_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[norm_weight_key] = step.updated_norm_weight
                        params[norm_bias_key] = step.updated_norm_bias
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
                        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    elif gpu_training_plan['kind'] == 'groupnorm_linear':
                        norm_node = gpu_training_plan['groupnorm_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        norm_weight_key = f'_w_{norm_node.name}'
                        norm_bias_key = f'_b_{norm_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_groupnorm_linear_training_step(
                            xb,
                            yb,
                            params[norm_weight_key],
                            params[norm_bias_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            num_groups=int(norm_node.attrs.get('num_groups', 1)),
                            norm_eps=float(norm_node.attrs.get('eps', 1e-5)),
                            norm_weight_velocity=velocity_state.get(norm_weight_key),
                            norm_bias_velocity=velocity_state.get(norm_bias_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[norm_weight_key] = step.updated_norm_weight
                        params[norm_bias_key] = step.updated_norm_bias
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
                        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    elif gpu_training_plan['kind'] == 'depthwise_layernorm2d_linear':
                        conv_node = gpu_training_plan['conv_node']
                        norm_node = gpu_training_plan['layernorm2d_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        conv_weight_key = f'_w_{conv_node.name}'
                        norm_weight_key = f'_w_{norm_node.name}'
                        norm_bias_key = f'_b_{norm_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_depthwise_layernorm2d_linear_training_step(
                            xb,
                            yb,
                            params[conv_weight_key],
                            params[norm_weight_key],
                            params[norm_bias_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
                            conv_weight_velocity=velocity_state.get(conv_weight_key),
                            norm_weight_velocity=velocity_state.get(norm_weight_key),
                            norm_bias_velocity=velocity_state.get(norm_bias_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[conv_weight_key] = step.updated_conv_weight
                        params[norm_weight_key] = step.updated_norm_weight
                        params[norm_bias_key] = step.updated_norm_bias
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[conv_weight_key] = step.updated_conv_weight_velocity
                        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
                        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    elif gpu_training_plan['kind'] == 'depthwise_layernorm2d_pointwise_linear':
                        depthwise_node = gpu_training_plan['depthwise_node']
                        norm_node = gpu_training_plan['layernorm2d_node']
                        pointwise_node = gpu_training_plan['pointwise_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        depthwise_weight_key = f'_w_{depthwise_node.name}'
                        norm_weight_key = f'_w_{norm_node.name}'
                        norm_bias_key = f'_b_{norm_node.name}'
                        pointwise_weight_key = f'_w_{pointwise_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_depthwise_layernorm2d_pointwise_linear_training_step(
                            xb,
                            yb,
                            params[depthwise_weight_key],
                            params[norm_weight_key],
                            params[norm_bias_key],
                            params[pointwise_weight_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
                            depthwise_weight_velocity=velocity_state.get(depthwise_weight_key),
                            norm_weight_velocity=velocity_state.get(norm_weight_key),
                            norm_bias_velocity=velocity_state.get(norm_bias_key),
                            pointwise_weight_velocity=velocity_state.get(pointwise_weight_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[depthwise_weight_key] = step.updated_depthwise_weight
                        params[norm_weight_key] = step.updated_norm_weight
                        params[norm_bias_key] = step.updated_norm_bias
                        params[pointwise_weight_key] = step.updated_pointwise_weight
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[depthwise_weight_key] = step.updated_depthwise_weight_velocity
                        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
                        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
                        velocity_state[pointwise_weight_key] = step.updated_pointwise_weight_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    elif gpu_training_plan['kind'] == 'depthwise_layernorm2d_pointwise_gelu_pointwise_linear':
                        depthwise_node = gpu_training_plan['depthwise_node']
                        norm_node = gpu_training_plan['layernorm2d_node']
                        pointwise1_node = gpu_training_plan['pointwise1_node']
                        pointwise2_node = gpu_training_plan['pointwise2_node']
                        gpu_linear_node = gpu_training_plan['linear_nodes'][0]
                        depthwise_weight_key = f'_w_{depthwise_node.name}'
                        norm_weight_key = f'_w_{norm_node.name}'
                        norm_bias_key = f'_b_{norm_node.name}'
                        pointwise1_weight_key = f'_w_{pointwise1_node.name}'
                        pointwise2_weight_key = f'_w_{pointwise2_node.name}'
                        linear_weight_key = f'_w_{gpu_linear_node.name}'
                        linear_bias_key = f'_b_{gpu_linear_node.name}'
                        step = native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
                            xb,
                            yb,
                            params[depthwise_weight_key],
                            params[norm_weight_key],
                            params[norm_bias_key],
                            params[pointwise1_weight_key],
                            params[pointwise2_weight_key],
                            params[linear_weight_key],
                            params[linear_bias_key],
                            lr=float(optimizer_view.lr),
                            momentum=float(ctx.momentum),
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            norm_eps=float(norm_node.attrs.get('eps', 1e-6)),
                            depthwise_weight_velocity=velocity_state.get(depthwise_weight_key),
                            norm_weight_velocity=velocity_state.get(norm_weight_key),
                            norm_bias_velocity=velocity_state.get(norm_bias_key),
                            pointwise1_weight_velocity=velocity_state.get(pointwise1_weight_key),
                            pointwise2_weight_velocity=velocity_state.get(pointwise2_weight_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[depthwise_weight_key] = step.updated_depthwise_weight
                        params[norm_weight_key] = step.updated_norm_weight
                        params[norm_bias_key] = step.updated_norm_bias
                        params[pointwise1_weight_key] = step.updated_pointwise1_weight
                        params[pointwise2_weight_key] = step.updated_pointwise2_weight
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[depthwise_weight_key] = step.updated_depthwise_weight_velocity
                        velocity_state[norm_weight_key] = step.updated_norm_weight_velocity
                        velocity_state[norm_bias_key] = step.updated_norm_bias_velocity
                        velocity_state[pointwise1_weight_key] = step.updated_pointwise1_weight_velocity
                        velocity_state[pointwise2_weight_key] = step.updated_pointwise2_weight_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
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
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            conv_weight_velocity=velocity_state.get(conv_weight_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            apply_relu_activation=bool(gpu_training_plan.get('apply_relu_activation', False)),
                            activation_kind=gpu_training_plan.get('activation_kind'),
                            activation_alpha=float(gpu_training_plan.get('activation_alpha', 0.01)),
                            apply_maxpool=bool(gpu_training_plan.get('apply_maxpool', False)),
                            conv_kind=str(gpu_training_plan.get('conv_kind', 'conv2d')),
                            bound_lib=ctx.device_runtime.bound_lib,
                        )
                        params = dict(params)
                        params[conv_weight_key] = step.updated_conv_weight
                        params[linear_weight_key] = step.updated_linear_weight
                        params[linear_bias_key] = step.updated_linear_bias
                        velocity_state[conv_weight_key] = step.updated_conv_weight_velocity
                        velocity_state[linear_weight_key] = step.updated_linear_weight_velocity
                        velocity_state[linear_bias_key] = step.updated_linear_bias_velocity
                    elif gpu_training_plan['kind'] == 'two_conv_relu_pool_linear':
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
                            grad_clip_value=float(ctx.grad_clip_global),
                            weight_decay=float(ctx.weight_decay),
                            label_smoothing=float(ctx.loss_cfg.get('label_smoothing', 0.0)),
                            conv1_weight_velocity=velocity_state.get(conv1_weight_key),
                            conv2_weight_velocity=velocity_state.get(conv2_weight_key),
                            linear_weight_velocity=velocity_state.get(linear_weight_key),
                            linear_bias_velocity=velocity_state.get(linear_bias_key),
                            activation_kind=gpu_training_plan.get('activation_kind'),
                            activation_alpha=float(gpu_training_plan.get('activation_alpha', 0.01)),
                            bound_lib=ctx.device_runtime.bound_lib,
                            return_intermediates=False,
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
                    else:
                        raise RuntimeError(
                            f"unhandled gpu_native training plan kind: {gpu_training_plan['kind']!r}"
                        )
                    loss_val = float(step.loss_mean)
                    batch_correct = int(step.correct_count)
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
                if batch_correct is not None:
                    running_correct += batch_correct
                    correct_seen += xb.shape[0]
                seen += xb.shape[0]
            train_loss = running_loss / max(seen, 1)
            train_acc = (running_correct / correct_seen) if correct_seen > 0 else None
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
            optimizer_runtime = optimizer_runtime_snapshot(optimizer_state)
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
                train_acc=train_acc,
                val_metrics=val_metrics,
                lr=float(optimizer_view.lr),
                epoch_time_s=epoch_time,
                amp_state=amp_epoch_state,
                optimizer_state=optimizer_epoch_state,
                planner_state=planner_epoch_state,
                device_runtime_state=ctx.device_runtime.summary(),
                support_tier_assessment=ctx.support_tier_assessment,
                execution_mode=ctx.execution_mode,
                selected_execution_mode=ctx.selected_execution_mode,
                tensor_execution_device=ctx.tensor_execution_device,
                execution_mode_policy=ctx.execution_mode_policy,
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
                    train_acc=train_acc,
                    val_acc=val_metrics['acc'],
                    lr=float(optimizer_view.lr),
                    epoch_time_s=epoch_time,
                    amp_state=amp_epoch_state,
                    optimizer_state=optimizer_epoch_state,
                )
            )

    amp_runtime = dict(optimizer_state.get('amp', {}))
    optimizer_runtime = optimizer_runtime_snapshot(optimizer_state)
    total_epoch_time = float(sum(epoch_times))
    epochs_completed = len(epoch_times)
    train_samples_per_epoch = int(ctx.x_train.shape[0])
    train_hotspot_profile = {}
    eval_hotspot_profile = {}
    if ctx.x_val.shape[0] > 0:
        eval_sample_batch = ctx.x_val[: min(ctx.batch_size, ctx.x_val.shape[0])]
        eval_hotspot_profile = profile_hotspots(
            ctx.graph,
            eval_sample_batch,
            best_params,
            amp_enabled=ctx.amp,
            mode='eval',
        )
    if ctx.x_train.shape[0] > 0:
        train_sample_batch = ctx.x_train[: min(ctx.batch_size, ctx.x_train.shape[0])]
        train_hotspot_profile = profile_hotspots(
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
        'hotspot_diff': build_hotspot_diff_summary(train_hotspot_profile, eval_hotspot_profile),
    }
    return best_params, best_val_acc, amp_runtime, optimizer_runtime, runtime_profile
