"""Data loading helpers shared by legacy CIFAR trainers."""
from __future__ import annotations

import numpy as np

from minicnn.data.cifar10 import load_cifar10, normalize_cifar
from minicnn.paths import DATA_ROOT


def load_normalized_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from minicnn.config import settings

    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10(
        str(DATA_ROOT),
        n_train=settings.N_TRAIN,
        n_val=settings.N_VAL,
        seed=settings.DATASET_SEED,
        train_batch_ids=settings.TRAIN_BATCH_IDS,
    )
    return (
        normalize_cifar(x_train),
        y_train,
        normalize_cifar(x_val),
        y_val,
        normalize_cifar(x_test),
        y_test,
    )
