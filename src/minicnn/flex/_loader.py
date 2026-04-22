from __future__ import annotations

import numpy as np

from minicnn.config.parsing import parse_bool


def define_augmented_tensor_dataset(torch, F, TensorDataset):
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

    return AugmentedTensorDataset


def make_loader(
    *,
    torch,
    DataLoader,
    dataset_cls,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    random_crop_padding: int = 0,
    horizontal_flip: bool = False,
    seed: int = 0,
):
    tx = torch.from_numpy(x.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.int64))
    dataset = dataset_cls(
        tx,
        ty,
        random_crop_padding=random_crop_padding,
        horizontal_flip=horizontal_flip,
        seed=seed,
    )
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
