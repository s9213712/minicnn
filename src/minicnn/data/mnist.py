"""MNIST download and loading utilities (pure Python / NumPy, no torchvision)."""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def _download_mnist(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        dest = data_root / filename
        if not dest.exists():
            url = f"{MNIST_URL}/{filename}"
            print(f"Downloading {url} …")
            urllib.request.urlretrieve(url, dest)


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad MNIST image magic: {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, h, w)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad MNIST label magic: {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def normalize_mnist(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1] then standardise with MNIST channel statistics."""
    return (x.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD


def load_mnist(
    data_root: str | Path,
    n_train: int = 60000,
    n_val: int = 10000,
    seed: int = 42,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_val, y_val, x_test, y_test) as uint8 NCHW arrays.

    x arrays have shape (N, 1, 28, 28) dtype uint8.
    y arrays have dtype int64.
    Call normalize_mnist() before feeding into a model.
    """
    data_root = Path(data_root)
    if download:
        _download_mnist(data_root)

    train_images = _read_images(data_root / MNIST_FILES["train_images"])
    train_labels = _read_labels(data_root / MNIST_FILES["train_labels"])
    test_images  = _read_images(data_root / MNIST_FILES["test_images"])
    test_labels  = _read_labels(data_root / MNIST_FILES["test_labels"])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_images))
    train_images = train_images[idx]
    train_labels = train_labels[idx]

    if n_train < 0 or n_val < 0:
        raise ValueError('n_train and n_val must be non-negative')
    if n_train + n_val > len(train_images):
        n_val = min(n_val, len(train_images))
        n_train = len(train_images) - n_val

    x_train = train_images[:n_train]
    y_train = train_labels[:n_train].astype(np.int64)
    x_val   = train_images[n_train:n_train + n_val]
    y_val   = train_labels[n_train:n_train + n_val].astype(np.int64)
    x_test  = test_images
    y_test  = test_labels.astype(np.int64)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_mnist_test(
    data_root: str | Path,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    data_root = Path(data_root)
    if download:
        _download_mnist(data_root)
    x = _read_images(data_root / MNIST_FILES["test_images"])
    y = _read_labels(data_root / MNIST_FILES["test_labels"]).astype(np.int64)
    return x, y
