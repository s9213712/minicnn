"""Small shared helpers for legacy training loops."""
from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class RunningMetrics:
    """Accumulates loss and accuracy counts across one epoch."""

    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0

    def update(self, loss_sum: float, correct: int, total: int) -> None:
        self.loss_sum += float(loss_sum)
        self.correct += int(correct)
        self.total += int(total)

    @property
    def loss(self) -> float:
        return self.loss_sum / max(self.total, 1)

    @property
    def acc_percent(self) -> float:
        return self.correct / max(self.total, 1) * 100.0


@dataclass
class LrState:
    """Layer-group learning rates used by the CUDA-compatible trainers."""

    conv1: float
    conv: float
    fc: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.conv1, self.conv, self.fc

    def reduce(self, factor: float, min_lr: float) -> bool:
        before = self.as_tuple()
        self.conv1 = max(self.conv1 * factor, min_lr)
        self.conv = max(self.conv * factor, min_lr)
        self.fc = max(self.fc * factor, min_lr)
        return self.as_tuple() != before


@dataclass
class FitState:
    """Tracks best validation score and no-improvement counters."""

    best_val_acc: float = -1.0
    best_epoch: int = -1
    epochs_no_improve: int = 0
    plateau_count: int = 0

    def observe(self, epoch: int, val_acc: float, min_delta: float) -> bool:
        improved = val_acc > (self.best_val_acc + min_delta)
        if improved:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.plateau_count = 0
            return True

        self.epochs_no_improve += 1
        self.plateau_count += 1
        return False

    def plateau_due(self, patience: int) -> bool:
        return self.plateau_count >= patience

    def reset_plateau(self) -> None:
        self.plateau_count = 0

    def should_stop(self, patience: int) -> bool:
        return self.epochs_no_improve >= patience


class EpochTimer:
    """Context manager that records elapsed wall time in seconds."""

    elapsed: float

    def __enter__(self) -> "EpochTimer":
        self._start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.elapsed = time.time() - self._start


def reduce_lr_on_plateau(
    fit: FitState,
    lr_state: LrState,
    patience: int,
    factor: float,
    min_lr: float,
) -> bool:
    if not fit.plateau_due(patience):
        return False
    changed = lr_state.reduce(factor, min_lr)
    fit.reset_plateau()
    return changed


def format_epoch_summary(
    epoch: int,
    epochs: int,
    metrics: RunningMetrics,
    val_acc: float,
    fit: FitState,
    lr_state: LrState,
    elapsed: float,
    save_msg: str,
    lr_separator: str = ", ",
) -> str:
    lrs = lr_separator.join(
        f"{lr:.6f}" for lr in (lr_state.conv1, lr_state.conv, lr_state.fc)
    )
    return (
        f"Epoch {epoch}/{epochs}: Loss={metrics.loss:.4f}, "
        f"Train={metrics.acc_percent:.2f}%, Val={val_acc:.2f}%, "
        f"BestVal={fit.best_val_acc:.2f}% @ {fit.best_epoch}, "
        f"LRs=({lrs}), Time={elapsed:.1f}s{save_msg}"
    )
