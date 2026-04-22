from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingRunState:
    best_val_acc: float = float('-inf')
    epochs_no_improve: int = 0
    periodic_checkpoints: list[str] = field(default_factory=list)


def step_scheduler(torch, scheduler, *, val_metrics: dict[str, float]) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_metrics['loss'])
    else:
        scheduler.step()


def handle_epoch_artifacts(
    torch,
    *,
    run_state: TrainingRunState,
    run_dir: Path,
    epoch: int,
    save_every_n_epochs: int,
    best_model_path: Path,
    checkpoint_path_for_epoch,
    model_state: dict[str, Any],
    val_acc: float,
    min_delta: float,
) -> bool:
    if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
        checkpoint_path = checkpoint_path_for_epoch(run_dir, epoch)
        torch.save({'epoch': epoch, 'model_state': model_state}, checkpoint_path)
        run_state.periodic_checkpoints.append(str(checkpoint_path))

    improved = val_acc > run_state.best_val_acc + min_delta
    if improved:
        run_state.best_val_acc = val_acc
        run_state.epochs_no_improve = 0
        torch.save({'model_state': model_state}, best_model_path)
    else:
        run_state.epochs_no_improve += 1
    return improved


def should_stop_early(run_state: TrainingRunState, *, early_stop_patience: int) -> bool:
    return early_stop_patience > 0 and run_state.epochs_no_improve >= early_stop_patience


def load_best_model_state(torch, best_model_path: Path, *, device):
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        checkpoint = torch.load(best_model_path, map_location=device)
    return checkpoint['model_state']
