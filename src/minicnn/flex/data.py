from __future__ import annotations

import numpy as np

from minicnn.config.parsing import parse_bool
from minicnn.flex._datasets import (
    DATASET_ARRAY_LOADERS as _DATASET_ARRAY_LOADERS_IMPL,
    load_custom_dataset_factory as _load_custom_dataset_factory_impl,
    load_dataset_arrays as _load_dataset_arrays_impl,
    load_test_arrays,
)

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
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers = int(train_cfg.get('num_workers', 0))
    seed = int(train_cfg.get('seed', dataset_cfg.get('seed', 42)))
    x_test, y_test = load_test_arrays(dataset_cfg, train_cfg)
    if x_test is None or y_test is None:
        return None
    return _make_loader(x_test, y_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed + 20_000)
