from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from minicnn.data.cifar10 import load_cifar10, load_cifar10_test, normalize_cifar
from minicnn.data.mnist import load_mnist, load_mnist_test, normalize_mnist
from minicnn.config.parsing import parse_bool

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = None if exc.name == 'torch' else exc
    F = None
    DataLoader = None
    TensorDataset = None
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
    F = None
    DataLoader = None
    TensorDataset = None


if TensorDataset is not None:
    class AugmentedTensorDataset(TensorDataset):
        def __init__(self, x, y, random_crop_padding: int = 0, horizontal_flip: bool = False, seed: int = 0):
            super().__init__(x, y)
            self.random_crop_padding = int(random_crop_padding)
            self.horizontal_flip = parse_bool(horizontal_flip, label='horizontal_flip')
            self.seed = int(seed)
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def _generator(self, index: int):
            worker = torch.utils.data.get_worker_info()
            worker_id = 0 if worker is None else worker.id
            # Large primes separate per-epoch, per-worker, per-sample seeds.
            seed = self.seed + self.epoch * 10_000_019 + worker_id * 1_000_003 + int(index)
            return torch.Generator().manual_seed(seed)

        def __getitem__(self, index):
            x, y = super().__getitem__(index)
            rng = self._generator(index)
            if self.random_crop_padding > 0:
                pad = self.random_crop_padding
                padded = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)
                top = int(torch.randint(0, 2 * pad + 1, (1,), generator=rng).item())
                left = int(torch.randint(0, 2 * pad + 1, (1,), generator=rng).item())
                x = padded[:, top:top + x.shape[-2], left:left + x.shape[-1]]
            if self.horizontal_flip and bool(torch.randint(0, 2, (1,), generator=rng).item()):
                x = torch.flip(x, dims=[-1])
            return x, y
else:  # pragma: no cover
    AugmentedTensorDataset = None


def _require_torch_data_support() -> None:
    if torch is None or DataLoader is None or AugmentedTensorDataset is None:
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                'PyTorch import failed in this environment for flex data loading. '
                f'{_TORCH_IMPORT_ERROR.__class__.__name__}: {_TORCH_IMPORT_ERROR}'
            )
        raise RuntimeError('PyTorch is required for train-flex')


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    random_crop_padding: int = 0,
    horizontal_flip: bool = False,
    seed: int = 0,
):
    _require_torch_data_support()
    tx = torch.from_numpy(x.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.int64))
    dataset = AugmentedTensorDataset(tx, ty, random_crop_padding=random_crop_padding, horizontal_flip=horizontal_flip, seed=seed)
    generator = torch.Generator().manual_seed(seed)

    def worker_init_fn(worker_id):
        torch.manual_seed(seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


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
        download=parse_bool(cfg.get('download', False), label='dataset.download'),
    )
    return normalize_cifar(x_train), y_train, normalize_cifar(x_val), y_val


def _mnist_dataset(cfg: dict, train_cfg: dict):
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


_DATASET_ARRAY_LOADERS = {
    'random': _random_dataset,
    'cifar10': _cifar_dataset,
    'mnist': _mnist_dataset,
}


def _load_custom_dataset_factory(factory_path: str):
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
    module = importlib.import_module(module_name)
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


def _load_dataset_arrays(dataset_cfg: dict, train_cfg: dict):
    dtype = dataset_cfg.get('type', 'cifar10')
    loader = _DATASET_ARRAY_LOADERS.get(dtype)
    if loader is not None:
        x_train, y_train, x_val, y_val = loader(dataset_cfg, train_cfg)
        return x_train, y_train, x_val, y_val, None, None

    factory = _load_custom_dataset_factory(str(dtype))
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


def create_dataloaders(dataset_cfg: dict, train_cfg: dict, augmentation_cfg: dict | None = None):
    _require_torch_data_support()
    x_train, y_train, x_val, y_val, _x_test, _y_test = _load_dataset_arrays(dataset_cfg, train_cfg)
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    seed = int(train_cfg.get('seed', dataset_cfg.get('seed', 42)))
    aug = augmentation_cfg or {}
    random_crop_padding = int(
        aug.get('random_crop_padding',
        dataset_cfg.get('random_crop_padding', train_cfg.get('random_crop_padding', 0)))
    )
    if parse_bool(aug.get('random_crop', False), label='augmentation.random_crop'):
        random_crop_padding = random_crop_padding or 4
    horizontal_flip = parse_bool(
        aug.get('horizontal_flip', dataset_cfg.get('horizontal_flip', train_cfg.get('horizontal_flip', False))),
        label='horizontal_flip',
    )
    train_loader = _make_loader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        random_crop_padding=random_crop_padding,
        horizontal_flip=horizontal_flip,
        seed=seed,
    )
    val_loader = _make_loader(x_val, y_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed + 10_000)
    return train_loader, val_loader


def create_test_dataloader(dataset_cfg: dict, train_cfg: dict):
    _require_torch_data_support()
    dtype = dataset_cfg.get('type', 'cifar10')
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    seed = int(train_cfg.get('seed', dataset_cfg.get('seed', 42)))
    if ':' in str(dtype):
        _x_train, _y_train, _x_val, _y_val, x_test, y_test = _load_dataset_arrays(dataset_cfg, train_cfg)
        if x_test is None or y_test is None:
            return None
        return _make_loader(x_test, y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed + 20_000)
    if dtype == 'cifar10':
        data_root = dataset_cfg.get('data_root', 'data/cifar-10-batches-py')
        x_test, y_test = load_cifar10_test(
            data_root=Path(data_root),
            download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
        )
        return _make_loader(normalize_cifar(x_test), y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed + 20_000)
    if dtype == 'mnist':
        data_root = dataset_cfg.get('data_root', 'data/mnist')
        x_test, y_test = load_mnist_test(
            data_root=Path(data_root),
            download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
        )
        return _make_loader(normalize_mnist(x_test), y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed + 20_000)
    return None
