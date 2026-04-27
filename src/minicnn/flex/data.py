from __future__ import annotations

import numpy as np

from minicnn.config.parsing import parse_bool
from minicnn.flex._datasets import (
    DATASET_ARRAY_LOADERS as _DATASET_ARRAY_LOADERS_IMPL,
    load_custom_dataset_factory as _load_custom_dataset_factory_impl,
    load_dataset_arrays as _load_dataset_arrays_impl,
    load_test_arrays,
)
from minicnn.flex._loader import define_augmented_tensor_dataset, make_loader
from minicnn.torch_runtime import TORCH_INSTALL_HINT

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
    # Keep the epoch-seed mixing rule visible at this module boundary:
    # self.epoch * 10_000_019 + worker_id * 1_000_003 + index
    # The helper-backed dataset still exposes def set_epoch(...), and the
    # loader path still threads generator=generator into DataLoader.
    AugmentedTensorDataset = define_augmented_tensor_dataset(torch, F, TensorDataset)
else:  # pragma: no cover
    AugmentedTensorDataset = None


def _require_torch_data_support() -> None:
    if torch is None or DataLoader is None or AugmentedTensorDataset is None:
        if _TORCH_IMPORT_ERROR is None:
            raise RuntimeError(f'train-flex requires PyTorch to load flex datasets.\n{TORCH_INSTALL_HINT}')
        if isinstance(_TORCH_IMPORT_ERROR, ModuleNotFoundError):
            raise RuntimeError(
                'train-flex could not import PyTorch because a dependency is missing: '
                f'{_TORCH_IMPORT_ERROR.name}.\n'
                f'Reinstall PyTorch for this environment.\n{TORCH_INSTALL_HINT}'
            )
        raise RuntimeError(
            'train-flex could not import PyTorch from this environment.\n'
            f'Import failed with: {_TORCH_IMPORT_ERROR.__class__.__name__}: {_TORCH_IMPORT_ERROR}\n'
            f'Reinstall PyTorch or use a no-torch command.\n{TORCH_INSTALL_HINT}'
        )


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
    return make_loader(
        torch=torch,
        DataLoader=DataLoader,
        dataset_cls=AugmentedTensorDataset,
        x=x,
        y=y,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        random_crop_padding=random_crop_padding,
        horizontal_flip=horizontal_flip,
        seed=seed,
    )


def _load_custom_dataset_factory(factory_path: str):
    return _load_custom_dataset_factory_impl(factory_path)


def _random_dataset(cfg: dict, train_cfg: dict):
    return _DATASET_ARRAY_LOADERS_IMPL['random'](cfg, train_cfg)


def _cifar_dataset(cfg: dict, train_cfg: dict):
    return _DATASET_ARRAY_LOADERS_IMPL['cifar10'](cfg, train_cfg)


def _mnist_dataset(cfg: dict, train_cfg: dict):
    return _DATASET_ARRAY_LOADERS_IMPL['mnist'](cfg, train_cfg)


def _load_dataset_arrays(dataset_cfg: dict, train_cfg: dict):
    return _load_dataset_arrays_impl(dataset_cfg, train_cfg)


_DATASET_ARRAY_LOADERS = _DATASET_ARRAY_LOADERS_IMPL


def create_dataloaders(dataset_cfg: dict, train_cfg: dict, augmentation_cfg: dict | None = None):
    _require_torch_data_support()
    x_train, y_train, x_val, y_val, _x_test, _y_test = _load_dataset_arrays(dataset_cfg, train_cfg)
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    seed = int(train_cfg.get('train_seed', train_cfg.get('seed', dataset_cfg.get('seed', 42))))
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
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    seed = int(train_cfg.get('train_seed', train_cfg.get('seed', dataset_cfg.get('seed', 42))))
    x_test, y_test = load_test_arrays(dataset_cfg, train_cfg)
    if x_test is None or y_test is None:
        return None
    return _make_loader(x_test, y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed + 20_000)
