from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.models import build_model_from_config
from minicnn.nn import Tensor, bce_with_logits_loss, cross_entropy, mse_loss, no_grad
from minicnn.optim import Adam, SGD
from minicnn.paths import BEST_MODELS_ROOT, DATA_ROOT


def _random_dataset(dataset_cfg: dict[str, Any]):
    input_shape = tuple(dataset_cfg.get('input_shape', [1, 4, 4]))
    num_classes = int(dataset_cfg.get('num_classes', 2))
    num_samples = int(dataset_cfg.get('num_samples', 16))
    val_samples = int(dataset_cfg.get('val_samples', 8))
    test_samples = int(dataset_cfg.get('test_samples', val_samples))
    seed = int(dataset_cfg.get('seed', 42))
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(num_samples, *input_shape)).astype(np.float32)
    y_train = rng.integers(0, num_classes, size=(num_samples,), dtype=np.int64)
    x_val = rng.normal(size=(val_samples, *input_shape)).astype(np.float32)
    y_val = rng.integers(0, num_classes, size=(val_samples,), dtype=np.int64)
    x_test = rng.normal(size=(test_samples, *input_shape)).astype(np.float32)
    y_test = rng.integers(0, num_classes, size=(test_samples,), dtype=np.int64)
    return x_train, y_train, x_val, y_val, x_test, y_test


def _cifar10_dataset(dataset_cfg: dict[str, Any]):
    from minicnn.data.cifar10 import load_cifar10, normalize_cifar
    data_root = str(dataset_cfg.get('data_root', DATA_ROOT))
    download = bool(dataset_cfg.get('download', False))
    n_train = int(dataset_cfg.get('num_samples', 45000))
    n_val = int(dataset_cfg.get('val_samples', 5000))
    seed = int(dataset_cfg.get('seed', 42))
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10(
        data_root, n_train=n_train, n_val=n_val, seed=seed, download=download,
    )
    return (
        normalize_cifar(x_train), y_train,
        normalize_cifar(x_val), y_val,
        normalize_cifar(x_test), y_test,
    )


def _mnist_dataset(dataset_cfg: dict[str, Any]):
    from minicnn.data.mnist import load_mnist, normalize_mnist
    data_root = str(dataset_cfg.get('data_root', DATA_ROOT / 'mnist'))
    download = bool(dataset_cfg.get('download', True))
    n_train = int(dataset_cfg.get('num_samples', 60000))
    n_val = int(dataset_cfg.get('val_samples', 5000))
    seed = int(dataset_cfg.get('seed', 42))
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist(
        data_root, n_train=n_train, n_val=n_val, seed=seed, download=download,
    )
    return (
        normalize_mnist(x_train), y_train,
        normalize_mnist(x_val), y_val,
        normalize_mnist(x_test), y_test,
    )


def _load_dataset(dataset_cfg: dict[str, Any]):
    dtype = str(dataset_cfg.get('type', 'random'))
    if dtype == 'cifar10':
        warnings.warn(
            'train-autograd with cifar10 uses NumPy Conv2d which is very slow. '
            'For fast CIFAR-10 training use train-flex or train-dual.',
            UserWarning,
            stacklevel=3,
        )
        return _cifar10_dataset(dataset_cfg)
    if dtype == 'mnist':
        warnings.warn(
            'train-autograd with mnist uses NumPy Conv2d which is slow for large datasets.',
            UserWarning,
            stacklevel=3,
        )
        return _mnist_dataset(dataset_cfg)
    if dtype == 'random':
        return _random_dataset(dataset_cfg)
    raise ValueError(f'train-autograd: unsupported dataset.type={dtype!r}; expected random, cifar10, or mnist')


def _make_optimizer(params, cfg: dict[str, Any]):
    optim_type = str(cfg.get('type', 'SGD'))
    lr = float(cfg.get('lr', 0.01))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    grad_clip = float(cfg.get('grad_clip', 0.0))
    if optim_type == 'Adam':
        return Adam(params, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip)
    if optim_type == 'SGD':
        return SGD(params, lr=lr, momentum=float(cfg.get('momentum', 0.0)), weight_decay=weight_decay, grad_clip=grad_clip)
    raise ValueError(
        f'train-autograd: unsupported optimizer.type={optim_type!r}; expected SGD or Adam'
    )


def _dense_targets(labels: np.ndarray, logits_shape: tuple[int, ...], loss_type: str) -> np.ndarray:
    if len(logits_shape) != 2:
        raise ValueError(f'{loss_type} expects 2D logits in train-autograd, got shape {logits_shape}')
    batch, outputs = logits_shape
    if labels.shape[0] != batch:
        raise ValueError(f'target batch size {labels.shape[0]} does not match logits batch size {batch}')
    if outputs == 1:
        return labels.reshape(batch, 1).astype(np.float32)
    dense = np.zeros((batch, outputs), dtype=np.float32)
    dense[np.arange(batch), labels.astype(np.int64)] = 1.0
    return dense


def _make_loss(loss_cfg: dict[str, Any]):
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    if loss_type == 'CrossEntropyLoss':
        return lambda logits, labels: cross_entropy(logits, labels)
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

    # Simple step-decay scheduler: multiply lr by gamma every step_size epochs.
    sched_enabled = sched_cfg.get('enabled', False)
    sched_step_size = int(sched_cfg.get('step_size', 10))
    sched_gamma = float(sched_cfg.get('gamma', 0.5))
    sched_min_lr = float(sched_cfg.get('min_lr', 1e-6))

    run_dir = create_run_dir(cfg)
    metrics_path = run_dir / 'metrics.jsonl'
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    best_path = BEST_MODELS_ROOT / f'{run_dir.name}_autograd_best.npz'
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

            if sched_enabled and epoch % sched_step_size == 0:
                new_lr = max(optimizer.lr * sched_gamma, sched_min_lr)
                optimizer.lr = new_lr

            row = {
                'epoch': epoch,
                'train_loss': total_loss / max(total, 1),
                'train_acc': last_train_acc,
                'val_acc': val_acc,
                'lr': optimizer.lr,
                'epoch_time_s': last_epoch_time,
            }
            metrics_file.write(json.dumps(row) + '\n')
            metrics_file.flush()
            print(
                f"Epoch {epoch}/{epochs}: loss={row['train_loss']:.4f}, "
                f"train_acc={last_train_acc * 100:.2f}%, "
                f"val_acc={val_acc * 100:.2f}%, "
                f"lr={optimizer.lr:.6g}, time={last_epoch_time:.1f}s"
            )
            if val_acc > best_val:
                best_val = val_acc
                np.savez(best_path, **model.state_dict())

    model.train(False)
    test_acc = _accuracy(model, x_test, y_test, batch_size, loss_type)
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
    return run_dir
