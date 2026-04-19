from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.models import build_model_from_config
from minicnn.nn import Tensor, cross_entropy, no_grad
from minicnn.optim import Adam, SGD
from minicnn.paths import BEST_MODELS_ROOT


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


def _make_optimizer(params, cfg: dict[str, Any]):
    optim_type = str(cfg.get('type', 'SGD'))
    lr = float(cfg.get('lr', 0.01))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    if optim_type == 'Adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    return SGD(params, lr=lr, momentum=float(cfg.get('momentum', 0.0)), weight_decay=weight_decay)


def _accuracy(model, x, y, batch_size: int) -> float:
    correct = 0
    total = 0
    with no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = Tensor(x[start:start + batch_size])
            logits = model(xb)
            pred = logits.data.argmax(axis=1)
            correct += int((pred == y[start:start + batch_size]).sum())
            total += pred.shape[0]
    return correct / max(total, 1)


def train_autograd_from_config(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get('dataset', {})
    if dataset_cfg.get('type', 'random') != 'random':
        raise ValueError('train-autograd currently supports dataset.type=random only')
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    x_train, y_train, x_val, y_val, x_test, y_test = _random_dataset(dataset_cfg)
    train_rng = np.random.default_rng(int(train_cfg.get('seed', 0)))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('seed', 42)))
    init_rng = np.random.default_rng(init_seed)
    model, final_shape = build_model_from_config(
        model_cfg,
        input_shape=dataset_cfg.get('input_shape', [1, 4, 4]),
        rng=init_rng,
    )
    optimizer = _make_optimizer(model.parameters(), cfg.get('optimizer', {'type': 'SGD', 'lr': 0.01}))
    batch_size = int(train_cfg.get('batch_size', 8))
    epochs = int(train_cfg.get('epochs', 1))
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
            for start in range(0, indices.shape[0], batch_size):
                idx = indices[start:start + batch_size]
                xb = Tensor(x_train[idx], requires_grad=False)
                yb = y_train[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.data) * len(idx)
                total += len(idx)
            last_epoch_time = time.perf_counter() - epoch_start
            last_train_acc = _accuracy(model, x_train, y_train, batch_size)
            val_acc = _accuracy(model, x_val, y_val, batch_size)
            row = {
                'epoch': epoch,
                'train_loss': total_loss / max(total, 1),
                'train_acc': last_train_acc,
                'val_acc': val_acc,
                'epoch_time_s': last_epoch_time,
            }
            metrics_file.write(json.dumps(row) + '\n')
            metrics_file.flush()
            if val_acc > best_val:
                best_val = val_acc
                np.savez(best_path, **model.state_dict())
    test_acc = _accuracy(model, x_test, y_test, batch_size)
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
