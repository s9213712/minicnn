from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.backward import BackwardExecutor
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
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
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    momentum: float
    grad_clip_global: float
    loss_type: str
    model_cfg: dict[str, Any]
    loss_cfg: dict[str, Any]
    scheduler_cfg: dict[str, Any]


def prepare_training_context(cfg: dict[str, Any], graph: NativeGraph) -> NativeTrainingContext:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    optim_cfg = cfg.get('optimizer', {})
    loss_cfg = cfg.get('loss', {})
    scheduler_cfg = cfg.get('scheduler', {})

    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    batch_size = int(train_cfg.get('batch_size', 64))
    epochs = int(train_cfg.get('epochs', 1))
    lr = float(optim_cfg.get('lr', 0.01))
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    momentum = float(optim_cfg.get('momentum', 0.0))
    grad_clip_global = float(optim_cfg.get('grad_clip_global', 0.0))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('seed', 42)))
    loss_type = _resolve_loss_type(loss_cfg)

    x_train, y_train, x_val, y_val = _load_numpy_data(cfg)
    params = _init_params(graph, seed=init_seed)

    return NativeTrainingContext(
        cfg=cfg,
        graph=graph,
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        input_shape=input_shape,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        grad_clip_global=grad_clip_global,
        loss_type=loss_type,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        scheduler_cfg=scheduler_cfg,
    )


def run_training_loop(
    ctx: NativeTrainingContext,
    *,
    run_dir: Path,
) -> tuple[dict[str, np.ndarray], float]:
    metrics_path = run_dir / 'metrics.jsonl'
    fwd = ForwardExecutor()
    bwd = BackwardExecutor()
    optimizer_view = SimpleNamespace(lr=ctx.lr)
    scheduler, scheduler_kind = _make_scheduler(ctx.scheduler_cfg, optimizer_view)
    optimizer_state: dict[str, Any] = {}
    best_val_acc = float('-inf')
    best_params = dict(ctx.params)
    params = ctx.params
    rng = np.random.default_rng(int(ctx.cfg.get('train', {}).get('init_seed', ctx.cfg.get('dataset', {}).get('seed', 42))))

    with metrics_path.open('w', encoding='utf-8') as mf:
        for epoch in range(1, ctx.epochs + 1):
            t0 = time.perf_counter()
            idx = rng.permutation(ctx.x_train.shape[0])
            x_shuf, y_shuf = ctx.x_train[idx], ctx.y_train[idx]
            running_loss = 0.0
            seen = 0
            for i in range(0, x_shuf.shape[0], ctx.batch_size):
                xb = x_shuf[i:i + ctx.batch_size]
                yb = y_shuf[i:i + ctx.batch_size]
                if xb.shape[0] == 0:
                    continue
                loss_val, params = train_step(
                    ctx.graph,
                    xb,
                    yb,
                    params,
                    lr=optimizer_view.lr,
                    loss_type=ctx.loss_type,
                    weight_decay=ctx.weight_decay,
                    momentum=ctx.momentum,
                    optimizer_state=optimizer_state,
                    grad_clip_global=ctx.grad_clip_global,
                    fwd_executor=fwd,
                    bwd_executor=bwd,
                )
                running_loss += loss_val * xb.shape[0]
                seen += xb.shape[0]
            train_loss = running_loss / max(seen, 1)
            val_metrics = _evaluate(ctx.graph, ctx.x_val, ctx.y_val, params, ctx.batch_size, ctx.loss_type)
            if scheduler is not None:
                if scheduler_kind == 'plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            epoch_time = time.perf_counter() - t0
            row = _build_epoch_row(
                epoch=epoch,
                train_loss=train_loss,
                val_metrics=val_metrics,
                lr=float(optimizer_view.lr),
                epoch_time_s=epoch_time,
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
                )
            )

    return best_params, best_val_acc


def finalize_training_run(
    ctx: NativeTrainingContext,
    *,
    run_dir: Path,
    best_params: dict[str, np.ndarray],
    best_val_acc: float,
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
        scheduler_cfg=ctx.scheduler_cfg,
        lr=ctx.lr,
        momentum=ctx.momentum,
        grad_clip_global=ctx.grad_clip_global,
        epochs=ctx.epochs,
        capabilities=capabilities,
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
    best_params, best_val_acc = run_training_loop(ctx, run_dir=run_dir)
    return finalize_training_run(
        ctx,
        run_dir=run_dir,
        best_params=best_params,
        best_val_acc=best_val_acc,
        capabilities=capabilities,
    )
