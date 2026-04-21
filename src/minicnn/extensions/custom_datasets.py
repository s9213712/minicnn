from __future__ import annotations

import numpy as np


def checkerboard_dataset(dataset_cfg: dict, train_cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    input_shape = tuple(dataset_cfg.get('input_shape', [1, 8, 8]))
    if len(input_shape) != 3:
        raise ValueError(f'checkerboard_dataset expects CHW input_shape, got {input_shape!r}')
    num_classes = int(dataset_cfg.get('num_classes', 2))
    if num_classes != 2:
        raise ValueError('checkerboard_dataset is a binary example and expects dataset.num_classes=2')

    seed = int(dataset_cfg.get('seed', train_cfg.get('seed', 42)))
    rng = np.random.default_rng(seed)

    def _split(count: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(loc=0.0, scale=0.1, size=(count, *input_shape)).astype(np.float32)
        y = rng.integers(0, 2, size=(count,), endpoint=False, dtype=np.int64)
        for index, label in enumerate(y):
            x[index, :, ::2, ::2] += 1.0 if label == 1 else -1.0
            x[index, :, 1::2, 1::2] += 1.0 if label == 1 else -1.0
        return x, y

    train_count = int(dataset_cfg.get('num_samples', 64))
    val_count = int(dataset_cfg.get('val_samples', 16))
    test_count = int(dataset_cfg.get('test_samples', val_count))
    return {
        'train': _split(train_count),
        'val': _split(val_count),
        'test': _split(test_count),
    }
