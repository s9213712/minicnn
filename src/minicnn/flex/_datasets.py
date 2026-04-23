from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from minicnn.config.parsing import parse_bool
from minicnn.data.cifar10 import load_cifar10, load_cifar10_test, normalize_cifar
from minicnn.data.mnist import load_mnist, load_mnist_test, normalize_mnist
from minicnn.user_errors import format_dataset_split_error, format_user_error


TRAIN_POOL_LIMITS = {
    'mnist': 60000,
    'cifar10': 50000,
}

TRAIN_POOL_EXAMPLES = {
    'mnist': (55000, 5000),
    'cifar10': (45000, 5000),
}


def _validate_named_dataset_split(cfg: dict) -> None:
    dataset_type = str(cfg.get('type', ''))
    limit = TRAIN_POOL_LIMITS.get(dataset_type)
    if limit is None:
        return
    num_samples = int(cfg.get('num_samples', 0))
    val_samples = int(cfg.get('val_samples', 0))
    if num_samples < 0 or val_samples < 0:
        example_num, example_val = TRAIN_POOL_EXAMPLES.get(dataset_type, (0, 0))
        raise ValueError(format_user_error(
            'Dataset split invalid',
            cause='num_samples and val_samples must be non-negative.',
            fix='Use values greater than or equal to 0.',
            example=f'num_samples={example_num}\nval_samples={example_val}',
        ))
    if num_samples + val_samples > limit:
        example_num, example_val = TRAIN_POOL_EXAMPLES.get(dataset_type, (0, 0))
        raise ValueError(format_dataset_split_error(
            dataset_name=dataset_type,
            train_pool_size=limit,
            num_samples=num_samples,
            val_samples=val_samples,
            example_num_samples=example_num,
            example_val_samples=example_val,
        ))


def _random_dataset(cfg: dict, _train_cfg: dict):
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


def _cifar_dataset(cfg: dict, _train_cfg: dict):
    _validate_named_dataset_split(cfg)
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
        download=parse_bool(cfg.get('download', False), label='dataset.download'),
    )
    return normalize_cifar(x_train), y_train, normalize_cifar(x_val), y_val


def _mnist_dataset(cfg: dict, _train_cfg: dict):
    _validate_named_dataset_split(cfg)
    data_root = cfg.get('data_root', 'data/mnist')
    n_train = int(cfg.get('num_samples', 60000))
    n_val = int(cfg.get('val_samples', 10000))
    seed = int(cfg.get('seed', 42))
    x_train, y_train, x_val, y_val, _x_test, _y_test = load_mnist(
        data_root=Path(data_root),
        n_train=n_train,
        n_val=n_val,
        seed=seed,
        download=parse_bool(cfg.get('download', False), label='dataset.download'),
    )
    return normalize_mnist(x_train), y_train, normalize_mnist(x_val), y_val


DATASET_ARRAY_LOADERS = {
    'random': _random_dataset,
    'cifar10': _cifar_dataset,
    'mnist': _mnist_dataset,
}


def load_custom_dataset_factory(factory_path: str):
    if ':' not in factory_path:
        raise ValueError(
            'Custom dataset.type must use dotted import syntax "package.module:factory", '
            f'got {factory_path!r}'
        )
    module_name, factory_name = factory_path.split(':', 1)
    if not module_name or not factory_name:
        raise ValueError(
            'Custom dataset.type must use dotted import syntax "package.module:factory", '
            f'got {factory_path!r}'
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(
                f'Custom dataset factory {factory_path!r} could not be imported: '
                f'module {module_name!r} was not found'
            ) from exc
        raise ValueError(
            f'Custom dataset factory {factory_path!r} failed while importing module '
            f'{module_name!r}: missing dependency {exc.name!r}'
        ) from exc
    except Exception as exc:
        raise ValueError(
            f'Custom dataset factory {factory_path!r} failed while importing module '
            f'{module_name!r}: {exc.__class__.__name__}: {exc}'
        ) from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError as exc:
        raise ValueError(
            f'Custom dataset factory {factory_path!r} could not be resolved: '
            f'{module_name!r} has no attribute {factory_name!r}'
        ) from exc
    if not callable(factory):
        raise ValueError(f'Custom dataset factory {factory_path!r} is not callable')
    return factory


def load_dataset_arrays(dataset_cfg: dict, train_cfg: dict):
    dtype = dataset_cfg.get('type', 'cifar10')
    loader = DATASET_ARRAY_LOADERS.get(dtype)
    if loader is not None:
        x_train, y_train, x_val, y_val = loader(dataset_cfg, train_cfg)
        return x_train, y_train, x_val, y_val, None, None

    factory = load_custom_dataset_factory(str(dtype))
    splits = factory(dataset_cfg, train_cfg)
    if not isinstance(splits, dict):
        raise ValueError(
            f'Custom dataset factory {dtype!r} must return a dict with train/val/test splits'
        )

    def _split(name: str):
        value = splits.get(name)
        if value is None:
            return None, None
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(
                f'Custom dataset factory {dtype!r} split {name!r} must be a tuple (x, y)'
            )
        x, y = value
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)

    x_train, y_train = _split('train')
    x_val, y_val = _split('val')
    x_test, y_test = _split('test')
    if x_train is None or y_train is None or x_val is None or y_val is None:
        raise ValueError(
            f'Custom dataset factory {dtype!r} must provide at least train and val splits'
        )
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_test_arrays(dataset_cfg: dict, train_cfg: dict):
    dtype = dataset_cfg.get('type', 'cifar10')
    if ':' in str(dtype):
        _x_train, _y_train, _x_val, _y_val, x_test, y_test = load_dataset_arrays(dataset_cfg, train_cfg)
        return x_test, y_test
    if dtype == 'cifar10':
        data_root = dataset_cfg.get('data_root', 'data/cifar-10-batches-py')
        x_test, y_test = load_cifar10_test(
            data_root=Path(data_root),
            download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
        )
        return normalize_cifar(x_test), y_test
    if dtype == 'mnist':
        data_root = dataset_cfg.get('data_root', 'data/mnist')
        x_test, y_test = load_mnist_test(
            data_root=Path(data_root),
            download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
        )
        return normalize_mnist(x_test), y_test
    return None, None
