from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minicnn.cuda_native.api import assess_cuda_native_support_tier, resolve_cuda_native_execution_mode
from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.planner import make_plan
from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.unified._cuda_native_context import NativeTrainingContext
from minicnn.unified._cuda_native_training_loop import run_training_loop
from minicnn.unified._cuda_native_bridge import (
    _best_checkpoint_path,
    _build_training_summary,
    _init_params,
    _load_numpy_data,
    _resolve_loss_type,
)

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
    execution_mode_info = resolve_cuda_native_execution_mode(cfg)
    selected_execution_mode = str(execution_mode_info.get('effective_execution_mode', 'reference_numpy'))
    requested_execution_mode = str(execution_mode_info.get('selected_execution_mode', selected_execution_mode))
    execution_mode_policy = {
        key: execution_mode_info[key]
        for key in (
            'fallback_execution_mode',
            'fallback_available',
            'fallback_active',
            'fallback_reason',
            'gpu_native_lowering_ready',
            'gpu_native_runtime_ready',
        )
        if key in execution_mode_info
    }
    tensor_execution_device = 'gpu' if selected_execution_mode == 'gpu_native' else 'cpu'
    bound_lib = None
    if selected_execution_mode == 'gpu_native':
        from minicnn.core._cuda_library import bind_symbols, ensure_cuda_runtime_available, load_library

        try:
            bound_lib = bind_symbols(load_library())
            ensure_cuda_runtime_available(bound_lib)
        except RuntimeError as exc:
            if requested_execution_mode != 'gpu_native_auto':
                raise
            selected_execution_mode = 'reference_numpy'
            tensor_execution_device = 'cpu'
            bound_lib = None
            execution_mode_policy = {
                **execution_mode_policy,
                'fallback_execution_mode': 'reference_numpy',
                'fallback_available': True,
                'fallback_active': True,
                'fallback_reason': str(exc),
                'gpu_native_runtime_ready': False,
            }
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
        selected_execution_mode=requested_execution_mode,
        tensor_execution_device=tensor_execution_device,
        execution_mode_policy=execution_mode_policy,
        device_runtime=device_runtime,
    )

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
        selected_execution_mode=ctx.selected_execution_mode,
        tensor_execution_device=ctx.tensor_execution_device,
        execution_mode_policy=ctx.execution_mode_policy,
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
