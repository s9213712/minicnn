"""CIFAR-10 loading and preprocessing helpers."""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from minicnn.user_errors import format_dataset_split_error, format_user_error


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIRNAME = "cifar-10-batches-py"
REQUIRED_FILES = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
)
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def _normalize_data_root(data_root) -> Path:
    return Path(data_root).expanduser()


def normalize_cifar(x):
    return ((x - CIFAR_MEAN) / CIFAR_STD).astype(np.float32)


def cifar10_ready(data_root):
    root = _normalize_data_root(data_root)
    return all((root / name).exists() for name in REQUIRED_FILES)


def _safe_extract(tar, path):
    base = Path(path).resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if base not in target.parents and target != base:
            raise RuntimeError(f"Unsafe path in CIFAR-10 archive: {member.name}")
    tar.extractall(path)


def prepare_cifar10(data_root, download=True):
    data_root = _normalize_data_root(data_root)
    if cifar10_ready(data_root):
        return data_root

    if not download:
        missing = [name for name in REQUIRED_FILES if not (data_root / name).exists()]
        raise FileNotFoundError(
            "CIFAR-10 Python batch files are missing:\n"
            f"  data_root={data_root}\n"
            f"  missing={missing}\n\n"
            "Prepare the dataset with:\n"
            "  minicnn prepare-data\n"
            "or pass an alternate path to train-flex with:\n"
            "  minicnn train-flex --config templates/cifar10/vgg_mini.yaml "
            "dataset.data_root=/path/to/cifar-10-batches-py\n"
            "or pass an alternate path to train-dual with:\n"
            "  minicnn train-dual --data-dir /path/to/cifar-10-batches-py ...\n"
            "or manually place the extracted cifar-10-batches-py directory under data/."
        )

    data_parent = data_root.parent
    data_parent.mkdir(parents=True, exist_ok=True)
    archive_path = data_parent / CIFAR10_ARCHIVE
    if not archive_path.exists():
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract(tar, data_parent)

    extracted_root = data_parent / CIFAR10_DIRNAME
    if extracted_root != data_root and extracted_root.exists() and not data_root.exists():
        extracted_root.rename(data_root)

    if not cifar10_ready(data_root):
        raise RuntimeError(f"CIFAR-10 preparation finished but required files are still missing in {data_root}")
    return data_root


def _load_batch(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIFAR-10 batch file not found: {path}")
    with open(path, "rb") as f:
        # CIFAR-10 python batches are legacy Python-2 pickles.  `latin1` keeps
        # NumPy payloads readable on modern Python without the NumPy 2.4
        # `align=0` visible deprecation warning seen with `encoding="bytes"`.
        batch = pickle.load(f, encoding="latin1")
    data = batch.get("data", batch.get(b"data"))
    labels = batch.get("labels", batch.get(b"labels"))
    if data is None or labels is None:
        raise ValueError(f"CIFAR-10 batch file has unexpected schema: {path}")
    x = (np.asarray(data, dtype=np.float32) / 255.0).reshape(-1, 3, 32, 32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def _load_training_batches(data_root, batch_ids):
    data_root = _normalize_data_root(data_root)
    x_parts = []
    y_parts = []
    for i in batch_ids:
        x_batch, y_batch = _load_batch(data_root / f"data_batch_{i}")
        x_parts.append(x_batch)
        y_parts.append(y_batch)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def load_cifar10(data_root, n_train=8000, n_val=2000, seed=None, train_batch_ids=(1,), download=True):
    data_root = prepare_cifar10(data_root, download=download)
    print(f"Loading CIFAR-10 training batches: {train_batch_ids}")
    x_train_all, y_train_all = _load_training_batches(data_root, train_batch_ids)
    print(f"Training samples: {x_train_all.shape[0]}")

    x_test, y_test = _load_batch(data_root / "test_batch")
    print(f"Test samples: {x_test.shape[0]}")

    if n_train < 0 or n_val < 0:
        raise ValueError(format_user_error(
            'Dataset split invalid',
            cause='num_samples and val_samples must be non-negative.',
            fix='Use values greater than or equal to 0.',
            example='num_samples=45000\nval_samples=5000',
        ))
    if n_train + n_val > x_train_all.shape[0]:
        raise ValueError(format_dataset_split_error(
            dataset_name='cifar10',
            train_pool_size=int(x_train_all.shape[0]),
            num_samples=n_train,
            val_samples=n_val,
            example_num_samples=45000,
            example_val_samples=5000,
        ))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(x_train_all.shape[0])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    return (
        x_train_all[train_idx],
        y_train_all[train_idx],
        x_train_all[val_idx],
        y_train_all[val_idx],
        x_test,
        y_test,
    )


def load_cifar10_test(data_root, download=True):
    data_root = prepare_cifar10(data_root, download=download)
    x_test, y_test = _load_batch(data_root / "test_batch")
    return x_test, y_test
