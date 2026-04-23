from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from minicnn.config.parsing import parse_bool
from minicnn.flex.runtime import create_run_dir
from minicnn.models import build_model_from_config
from minicnn.nn import Tensor, bce_with_logits_loss, cross_entropy, mse_loss, no_grad
from minicnn.optim import Adam, AdamW, RMSprop, SGD
from minicnn.schedulers.cosine import CosineAnnealingLR
from minicnn.schedulers.plateau import ReduceLROnPlateau
from minicnn.schedulers.step import StepLR
from minicnn.training._autograd_data import (
    _cifar10_dataset as _cifar10_dataset_impl,
    _mnist_dataset as _mnist_dataset_impl,
    load_autograd_dataset,
)
from minicnn.training._autograd_reporting import (
    build_epoch_row,
    dump_autograd_summary,
    reload_best_model,
    resolve_autograd_artifacts,
    save_best_model,
    write_epoch_row,
)
from minicnn.training.events import emit_training_event


def _cifar10_dataset(dataset_cfg: dict[str, Any]):
    normalized = dict(dataset_cfg)
    normalized['download'] = parse_bool(normalized.get('download', False), label='dataset.download')
    return _cifar10_dataset_impl(normalized)


def _mnist_dataset(dataset_cfg: dict[str, Any]):
    return _mnist_dataset_impl(dataset_cfg)


def _load_dataset(dataset_cfg: dict[str, Any]):
    return load_autograd_dataset(dataset_cfg)


def _make_optimizer(params, cfg: dict[str, Any]):
    optim_type = str(cfg.get('type', 'SGD'))
    lr = float(cfg.get('lr', 0.01))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    grad_clip = float(cfg.get('grad_clip', 0.0))
    if optim_type in ('Adam', 'adam'):
        return Adam(params, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip)
    if optim_type in ('AdamW', 'adamw'):
        return AdamW(params, lr=lr, weight_decay=weight_decay if weight_decay else 0.01, grad_clip=grad_clip)
    if optim_type in ('RMSprop', 'rmsprop'):
        return RMSprop(params, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip, momentum=float(cfg.get('momentum', 0.0)))
    if optim_type in ('SGD', 'sgd'):
        return SGD(params, lr=lr, momentum=float(cfg.get('momentum', 0.0)), weight_decay=weight_decay, grad_clip=grad_clip)
    raise ValueError(
        f'train-autograd: unsupported optimizer.type={optim_type!r}; expected SGD, Adam, AdamW, or RMSprop'
    )


def _make_scheduler(optimizer, cfg: dict[str, Any]):
    sched_enabled = parse_bool(cfg.get('enabled', False), label='scheduler.enabled')
    if not sched_enabled:
        return None
    sched_type = str(cfg.get('type', 'step'))
    if sched_type == 'none':
        return None
    if sched_type == 'step':
        step_size = int(cfg.get('step_size', 10))
        gamma = float(cfg.get('gamma', 0.5))
        min_lr = float(cfg.get('min_lr', 1e-6))
        if step_size < 1:
            raise ValueError(f'scheduler.step_size must be >= 1, got {step_size}')
        if gamma <= 0:
            raise ValueError(f'scheduler.gamma must be > 0, got {gamma}')
        return StepLR(optimizer, step_size=step_size, gamma=gamma, min_lr=min_lr)
    if sched_type == 'cosine':
        T_max = int(cfg.get('T_max', cfg.get('t_max', 10)))
        lr_min = float(cfg.get('lr_min', cfg.get('min_lr', 0.0)))
        return CosineAnnealingLR(optimizer, T_max=T_max, lr_min=lr_min)
    if sched_type == 'plateau':
        factor = float(cfg.get('factor', 0.5))
        patience = int(cfg.get('patience', 3))
        min_lr = float(cfg.get('min_lr', 1e-5))
        return ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr)
    raise ValueError(f'train-autograd: unsupported scheduler.type={sched_type!r}; expected none, step, cosine, or plateau')


def _dense_targets(labels: np.ndarray, logits_shape: tuple[int, ...], loss_type: str) -> np.ndarray:
    if len(logits_shape) != 2:
        raise ValueError(f'{loss_type} expects 2D logits in train-autograd, got shape {logits_shape}')
    batch, outputs = logits_shape
    if labels.shape[0] != batch:
        raise ValueError(f'target batch size {labels.shape[0]} does not match logits batch size {batch}')
    if outputs == 1:
        if loss_type == 'BCEWithLogitsLoss':
            bad = labels[(labels < 0) | (labels > 1)]
            if len(bad) > 0:
                raise ValueError(
                    f'BCEWithLogitsLoss binary classification labels must be in {{0, 1}}, '
                    f'but got values including {sorted(set(int(v) for v in bad[:5]))}. '
                    'Use CrossEntropyLoss for multi-class classification.'
                )
        return labels.reshape(batch, 1).astype(np.float32)
    dense = np.zeros((batch, outputs), dtype=np.float32)
    dense[np.arange(batch), labels.astype(np.int64)] = 1.0
    return dense


def _make_loss(loss_cfg: dict[str, Any]):
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    if loss_type == 'CrossEntropyLoss':
        smoothing = float(loss_cfg.get('label_smoothing', 0.0))
        return lambda logits, labels: cross_entropy(logits, labels, label_smoothing=smoothing)
    if loss_type == 'MSELoss':
        return lambda logits, labels: mse_loss(logits, _dense_targets(labels, logits.data.shape, loss_type))
    if loss_type == 'BCEWithLogitsLoss':
        return lambda logits, labels: bce_with_logits_loss(logits, _dense_targets(labels, logits.data.shape, loss_type))
    raise ValueError(
        "train-autograd supports loss.type values: CrossEntropyLoss, MSELoss, BCEWithLogitsLoss"
    )


def _accuracy(model, x, y, batch_size: int, loss_type: str = 'CrossEntropyLoss') -> float:
    correct = 0
    total = 0
    with no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = Tensor(x[start:start + batch_size])
            yb = y[start:start + batch_size]
            logits = model(xb)
            if loss_type == 'BCEWithLogitsLoss' and logits.data.shape[1] == 1:
                # binary: positive when logit >= 0
                pred = (logits.data[:, 0] >= 0.0).astype(np.int64)
            else:
                pred = logits.data.argmax(axis=1)
            correct += int((pred == yb).sum())
            total += pred.shape[0]
    return correct / max(total, 1)


def train_autograd_from_config(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    sched_cfg = cfg.get('scheduler', {})

    if parse_bool(train_cfg.get('amp', False), label='train.amp'):
        raise ValueError(
            'train-autograd (NumPy backend) does not support amp=true; '
            'use engine.backend=torch for mixed-precision training'
        )

    x_train, y_train, x_val, y_val, x_test, y_test = _load_dataset(dataset_cfg)
    train_rng = np.random.default_rng(int(train_cfg.get('seed', 0)))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('seed', 42)))
    init_rng = np.random.default_rng(init_seed)
    model, final_shape = build_model_from_config(
        model_cfg,
        input_shape=dataset_cfg.get('input_shape', [1, 4, 4]),
        rng=init_rng,
    )
    optim_cfg = cfg.get('optimizer', {'type': 'SGD', 'lr': 0.01})
    optimizer = _make_optimizer(model.parameters(), optim_cfg)
    loss_cfg = cfg.get('loss', {'type': 'CrossEntropyLoss'})
    loss_fn = _make_loss(loss_cfg)
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    batch_size = int(train_cfg.get('batch_size', 8))
    epochs = int(train_cfg.get('epochs', 1))
    if epochs < 1:
        raise ValueError(f'train.epochs must be >= 1, got {epochs}')

    scheduler = _make_scheduler(optimizer, sched_cfg)

    run_dir = create_run_dir(cfg)
    metrics_path, best_path = resolve_autograd_artifacts(run_dir)
    best_val = -1.0
    last_train_acc = 0.0
    last_epoch_time = 0.0

    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            indices = train_rng.permutation(x_train.shape[0])
            total_loss = 0.0
            total = 0
            model.train(True)
            for start in range(0, indices.shape[0], batch_size):
                idx = indices[start:start + batch_size]
                xb = Tensor(x_train[idx], requires_grad=False)
                yb = y_train[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.data) * len(idx)
                total += len(idx)

            last_epoch_time = time.perf_counter() - epoch_start
            model.train(False)
            last_train_acc = _accuracy(model, x_train, y_train, batch_size, loss_type)
            val_acc = _accuracy(model, x_val, y_val, batch_size, loss_type)
            row = build_epoch_row(
                epoch=epoch,
                train_loss=total_loss / max(total, 1),
                train_acc=last_train_acc,
                val_acc=val_acc,
                lr=optimizer.lr,
                epoch_time_s=last_epoch_time,
            )

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metric=row['train_loss'])
                else:
                    scheduler.step()

            write_epoch_row(metrics_file, row)
            emit_training_event(
                'epoch_summary',
                {
                    'epoch': epoch,
                    'epochs': epochs,
                    'train_metrics': {'loss': row['train_loss'], 'acc': last_train_acc},
                    'val_metrics': {'loss': row['train_loss'], 'acc': val_acc},
                    'lr': optimizer.lr,
                    'epoch_time_s': last_epoch_time,
                    'saved_best': val_acc > best_val,
                },
            )
            if val_acc > best_val:
                best_val = val_acc
                save_best_model(best_path, model)

    # Reload best checkpoint before final test evaluation so test_acc matches best_val_acc.
    reload_best_model(best_path, model)
    model.train(False)
    test_acc = _accuracy(model, x_test, y_test, batch_size, loss_type)
    dump_autograd_summary(
        run_dir=run_dir,
        dataset_cfg=dataset_cfg,
        final_shape=final_shape,
        best_path=best_path,
        last_train_acc=last_train_acc,
        best_val=best_val,
        test_acc=test_acc,
        last_epoch_time=last_epoch_time,
    )
    return run_dir
