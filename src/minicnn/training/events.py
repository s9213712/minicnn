from __future__ import annotations

from typing import Any


def format_epoch_summary_event(
    *,
    epoch: int,
    epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    epoch_time_s: float,
    saved_best: bool,
) -> str:
    save_msg = ' saved_best' if saved_best else ''
    return (
        f"Epoch {epoch}/{epochs}: loss={train_metrics['loss']:.4f}, "
        f"train_acc={train_metrics['acc'] * 100:.2f}%, "
        f"val_acc={val_metrics['acc'] * 100:.2f}%, "
        f"lr={lr:.6g}, "
        f"time={epoch_time_s:.1f}s{save_msg}"
    )


def format_early_stop_event(*, epoch: int, best_val_acc: float) -> str:
    return (
        f"Early stopping after {epoch} epochs; "
        f"best val_acc={best_val_acc * 100:.2f}%."
    )


def emit_training_event(
    event: str,
    payload: dict[str, Any],
    *,
    writer=print,
) -> str:
    if event == 'epoch_summary':
        message = format_epoch_summary_event(
            epoch=int(payload['epoch']),
            epochs=int(payload['epochs']),
            train_metrics=payload['train_metrics'],
            val_metrics=payload['val_metrics'],
            lr=float(payload['lr']),
            epoch_time_s=float(payload['epoch_time_s']),
            saved_best=bool(payload.get('saved_best', False)),
        )
    elif event == 'early_stop':
        message = format_early_stop_event(
            epoch=int(payload['epoch']),
            best_val_acc=float(payload['best_val_acc']),
        )
    else:
        raise ValueError(f'Unknown training event {event!r}')
    writer(message)
    return message
