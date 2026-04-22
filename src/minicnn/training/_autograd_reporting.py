from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from minicnn.flex.runtime import dump_summary
from minicnn.paths import BEST_MODELS_ROOT


def resolve_autograd_artifacts(run_dir: Path) -> tuple[Path, Path]:
    metrics_path = run_dir / 'metrics.jsonl'
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    best_path = BEST_MODELS_ROOT / f'{run_dir.name}_autograd_best.npz'
    return metrics_path, best_path


def build_epoch_row(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_acc: float,
    lr: float,
    epoch_time_s: float,
) -> dict[str, Any]:
    return {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lr': lr,
        'epoch_time_s': epoch_time_s,
    }


def write_epoch_row(metrics_file, row: dict[str, Any]) -> None:
    metrics_file.write(json.dumps(row) + '\n')
    metrics_file.flush()


def save_best_model(best_path: Path, model) -> None:
    np.savez(best_path, **model.state_dict())


def reload_best_model(best_path: Path, model) -> None:
    if not best_path.exists():
        return
    ckpt = np.load(best_path)
    model.load_state_dict({k: ckpt[k] for k in ckpt.files})


def dump_autograd_summary(
    run_dir: Path,
    dataset_cfg: dict[str, Any],
    final_shape,
    best_path: Path,
    last_train_acc: float,
    best_val: float,
    test_acc: float,
    last_epoch_time: float,
) -> None:
    dump_summary(run_dir, {
        'effective_backend': 'autograd',
        'run_dir': str(run_dir),
        'best_model_path': str(best_path),
        'input_shape': list(dataset_cfg.get('input_shape', [1, 4, 4])),
        'final_shape': list(final_shape),
        'train_acc': last_train_acc,
        'best_val_acc': best_val,
        'test_acc': test_acc,
        'epoch_time_s': last_epoch_time,
    })
