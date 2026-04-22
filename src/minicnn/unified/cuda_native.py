"""Bridge: shared unified config → cuda_native backend.

Responsibilities:
- Validate the config against cuda_native constraints (clear errors, no silent fallback)
- Translate config layers into a NativeGraph
- Initialize parameters
- Load dataset as numpy arrays (no torch dependency)
- Run minimal training loop and return a summary dict
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.api import (
    build_cuda_native_graph,
    get_capability_summary,
    validate_cuda_native_config,
)
from minicnn.cuda_native.backward import BackwardExecutor
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.training import sgd_update, train_step
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


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def check_config(cfg: dict[str, Any]) -> list[str]:
    """Return validation errors for *cfg* against cuda_native constraints."""
    return validate_cuda_native_config(cfg)


def get_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_capability_summary()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_cuda_native_training(cfg: dict[str, Any]) -> Path:
    """Run a full training loop for cuda_native and return the run directory.

    This is the backend-specific training entry point wired into
    unified/trainer.py when engine.backend = cuda_native.
    """
    errors = check_config(cfg)
    if errors:
        raise ValueError(
            'Config is not compatible with cuda_native:\n- ' + '\n- '.join(errors)
        )

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
    graph = build_cuda_native_graph(model_cfg, (batch_size, *input_shape))
    params = _init_params(graph, seed=init_seed)
    run_dir = create_run_dir(cfg)
    metrics_path = run_dir / 'metrics.jsonl'

    fwd = ForwardExecutor()
    bwd = BackwardExecutor()
    optimizer_view = SimpleNamespace(lr=lr)
    scheduler, scheduler_kind = _make_scheduler(scheduler_cfg, optimizer_view)
    optimizer_state: dict[str, Any] = {}
    best_val_acc = float('-inf')
    best_params = dict(params)
    rng = np.random.default_rng(init_seed)

    with metrics_path.open('w', encoding='utf-8') as mf:
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            idx = rng.permutation(x_train.shape[0])
            x_shuf, y_shuf = x_train[idx], y_train[idx]
            running_loss = 0.0
            seen = 0
            for i in range(0, x_shuf.shape[0], batch_size):
                xb = x_shuf[i:i + batch_size]
                yb = y_shuf[i:i + batch_size]
                if xb.shape[0] == 0:
                    continue
                loss_val, params = train_step(
                    graph, xb, yb, params, lr=optimizer_view.lr,
                    loss_type=loss_type, weight_decay=weight_decay,
                    momentum=momentum,
                    optimizer_state=optimizer_state,
                    grad_clip_global=grad_clip_global,
                    fwd_executor=fwd, bwd_executor=bwd,
                )
                running_loss += loss_val * xb.shape[0]
                seen += xb.shape[0]
            train_loss = running_loss / max(seen, 1)
            val_metrics = _evaluate(graph, x_val, y_val, params, batch_size, loss_type)
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
                    epochs=epochs,
                    train_loss=train_loss,
                    val_acc=val_metrics['acc'],
                    lr=float(optimizer_view.lr),
                    epoch_time_s=epoch_time,
                )
            )

    # Save best weights
    best_path = _best_checkpoint_path(run_dir)
    np.savez(str(best_path), **best_params)

    summary = _build_training_summary(
        run_dir=run_dir,
        best_path=best_path,
        best_val_acc=best_val_acc,
        input_shape=input_shape,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        scheduler_cfg=scheduler_cfg,
        lr=lr,
        momentum=momentum,
        grad_clip_global=grad_clip_global,
        epochs=epochs,
        capabilities=get_capability_summary(),
    )
    dump_summary(run_dir, summary)
    return run_dir
