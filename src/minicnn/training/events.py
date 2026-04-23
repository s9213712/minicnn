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


def format_batch_progress_event(*, batch_idx: int, num_batches: int, loss: float, acc_percent: float) -> str:
    return (
        f"  Batch {batch_idx}/{num_batches}: "
        f"loss={loss:.4f}, "
        f"acc={acc_percent:.1f}%"
    )


def format_lr_reduced_event(*, conv1: float, conv: float, fc: float, label: str = 'LR reduced') -> str:
    return (
        f"  {label} -> conv1={conv1:.6f}, "
        f"conv={conv:.6f}, fc={fc:.6f}"
    )


def format_legacy_early_stop_event(*, epoch: int, best_val_acc: float, best_epoch: int) -> str:
    return (
        f"Early stopping after {epoch} epochs; "
        f"best val {best_val_acc:.2f}% at epoch {best_epoch}."
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
    elif event == 'batch_progress':
        message = format_batch_progress_event(
            batch_idx=int(payload['batch_idx']),
            num_batches=int(payload['num_batches']),
            loss=float(payload['loss']),
            acc_percent=float(payload['acc_percent']),
        )
    elif event == 'lr_reduced':
        message = format_lr_reduced_event(
            conv1=float(payload['conv1']),
            conv=float(payload['conv']),
            fc=float(payload['fc']),
            label=str(payload.get('label', 'LR reduced')),
        )
    elif event == 'legacy_early_stop':
        message = format_legacy_early_stop_event(
            epoch=int(payload['epoch']),
            best_val_acc=float(payload['best_val_acc']),
            best_epoch=int(payload['best_epoch']),
        )
    else:
        raise ValueError(f'Unknown training event {event!r}')
    writer(message)
    return message
