from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from minicnn.config.parsing import parse_bool
from minicnn.paths import DATA_ROOT


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
    download = parse_bool(dataset_cfg.get('download', False), label='dataset.download')
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
    download = parse_bool(dataset_cfg.get('download', True), label='dataset.download')
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


def load_autograd_dataset(dataset_cfg: dict[str, Any]):
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
