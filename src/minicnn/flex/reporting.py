from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minicnn.paths import BEST_MODELS_ROOT
from minicnn.training.events import (
    emit_training_event as _emit_training_event_impl,
    format_epoch_summary_event,
)

TRAINING_SUMMARY_SCHEMA_VERSION = 1


def _best_model_path(run_dir: Path) -> Path:
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return BEST_MODELS_ROOT / f'{run_dir.name}_best.pt'


def _checkpoint_path(run_dir: Path, epoch: int) -> Path:
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return BEST_MODELS_ROOT / f'{run_dir.name}_epoch_{epoch}.pt'


def _build_epoch_row(
    *,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    epoch_time_s: float,
) -> dict[str, Any]:
    return {
        'epoch': epoch,
        'train_loss': train_metrics['loss'],
        'train_acc': train_metrics['acc'],
        'val_loss': val_metrics['loss'],
        'val_acc': val_metrics['acc'],
        'lr': lr,
        'epoch_time_s': epoch_time_s,
    }


def _write_metrics_row(metrics_file, row: dict[str, Any]) -> None:
    metrics_file.write(json.dumps(row) + '\n')
    metrics_file.flush()


def _epoch_log_message(
    *,
    epoch: int,
    epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    epoch_time_s: float,
    saved_best: bool,
) -> str:
    return format_epoch_summary_event(
        epoch=epoch,
        epochs=epochs,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lr=lr,
        epoch_time_s=epoch_time_s,
        saved_best=saved_best,
    )


def emit_training_event(
    event: str,
    payload: dict[str, Any],
    *,
    writer=print,
) -> str:
    return _emit_training_event_impl(event, payload, writer=writer)


def _build_training_summary(
    *,
    device,
    run_dir: Path,
    best_model_path: Path,
    input_shape: tuple[int, ...],
    model_cfg: dict[str, Any],
    cfg: dict[str, Any],
    periodic_checkpoints: list[str],
    test_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    summary = {
        'schema_version': TRAINING_SUMMARY_SCHEMA_VERSION,
        'artifact_kind': 'training_run_summary',
        'status': 'ok',
        'selected_backend': 'torch',
        'effective_backend': 'torch',
        'variant': str(device),
        'device': str(device),
        'run_dir': str(run_dir),
        'best_model_path': str(best_model_path),
        'input_shape': list(input_shape),
        'model_layers': [layer.get('type') for layer in model_cfg.get('layers', [])],
        'optimizer': cfg.get('optimizer', {}).get('type'),
        'loss': cfg.get('loss', {}).get('type'),
        'scheduler': cfg.get('scheduler', {}).get('type') if cfg.get('scheduler', {}).get('enabled') else None,
        'periodic_checkpoints': list(periodic_checkpoints),
        'test_loss': None,
        'test_acc': None,
    }
    if test_metrics is not None:
        summary['test_loss'] = test_metrics['loss']
        summary['test_acc'] = test_metrics['acc']
    return summary
