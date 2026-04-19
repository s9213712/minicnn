from __future__ import annotations

from pathlib import Path

import numpy as np

from minicnn.data.cifar10 import load_cifar10, load_cifar10_test, normalize_cifar

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None
    TensorDataset = None


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 0):
    tx = torch.from_numpy(x.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.int64))
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _random_dataset(cfg: dict, train_cfg: dict):
    input_shape = tuple(cfg.get('input_shape', [3, 32, 32]))
    num_classes = int(cfg.get('num_classes', 10))
    num_samples = int(cfg.get('num_samples', 512))
    val_samples = int(cfg.get('val_samples', max(64, num_samples // 4)))
    seed = int(cfg.get('seed', 42))
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(num_samples, *input_shape)).astype(np.float32)
    y_train = rng.integers(0, num_classes, size=(num_samples,), endpoint=False, dtype=np.int64)
    x_val = rng.normal(size=(val_samples, *input_shape)).astype(np.float32)
    y_val = rng.integers(0, num_classes, size=(val_samples,), endpoint=False, dtype=np.int64)
    return x_train, y_train, x_val, y_val


def _cifar_dataset(cfg: dict, train_cfg: dict):
    data_root = cfg.get('data_root', 'data/cifar-10-batches-py')
    n_train = int(cfg.get('num_samples', 512))
    n_val = int(cfg.get('val_samples', 128))
    seed = int(cfg.get('seed', 42))
    x_train, y_train, x_val, y_val, _x_test, _y_test = load_cifar10(
        data_root=Path(data_root),
        n_train=n_train,
        n_val=n_val,
        seed=seed,
        train_batch_ids=(1, 2, 3, 4, 5),
        download=bool(cfg.get('download', False)),
    )
    return normalize_cifar(x_train), y_train, normalize_cifar(x_val), y_val


def create_dataloaders(dataset_cfg: dict, train_cfg: dict):
    if torch is None:
        raise RuntimeError('PyTorch is required for train-flex')
    dtype = dataset_cfg.get('type', 'cifar10')
    if dtype == 'random':
        x_train, y_train, x_val, y_val = _random_dataset(dataset_cfg, train_cfg)
    elif dtype == 'cifar10':
        x_train, y_train, x_val, y_val = _cifar_dataset(dataset_cfg, train_cfg)
    else:
        raise ValueError(f'Unsupported dataset.type: {dtype}')
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    train_loader = _make_loader(x_train, y_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _make_loader(x_val, y_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def create_test_dataloader(dataset_cfg: dict, train_cfg: dict):
    if torch is None:
        raise RuntimeError('PyTorch is required for train-flex')
    if dataset_cfg.get('type', 'cifar10') != 'cifar10':
        return None
    data_root = dataset_cfg.get('data_root', 'data/cifar-10-batches-py')
    x_test, y_test = load_cifar10_test(data_root=Path(data_root), download=bool(dataset_cfg.get('download', False)))
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    return _make_loader(normalize_cifar(x_test), y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
