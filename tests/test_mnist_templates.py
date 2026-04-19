from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from minicnn.flex.builder import build_model
from minicnn.flex.config import load_flex_config


def _write_mnist_images(path: Path, images: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n, _c, h, w = images.shape
    with gzip.open(path, 'wb') as fh:
        fh.write(struct.pack('>IIII', 2051, n, h, w))
        fh.write(images.reshape(n, h, w).astype(np.uint8).tobytes())


def _write_mnist_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as fh:
        fh.write(struct.pack('>II', 2049, labels.shape[0]))
        fh.write(labels.astype(np.uint8).tobytes())


def _write_fake_mnist(root: Path) -> None:
    from minicnn.data.mnist import MNIST_FILES

    train_x = np.arange(4 * 28 * 28, dtype=np.uint8).reshape(4, 1, 28, 28)
    test_x = np.arange(3 * 28 * 28, dtype=np.uint8).reshape(3, 1, 28, 28)
    _write_mnist_images(root / MNIST_FILES['train_images'], train_x)
    _write_mnist_labels(root / MNIST_FILES['train_labels'], np.array([0, 1, 2, 3], dtype=np.uint8))
    _write_mnist_images(root / MNIST_FILES['test_images'], test_x)
    _write_mnist_labels(root / MNIST_FILES['test_labels'], np.array([4, 5, 6], dtype=np.uint8))


def test_load_mnist_reads_idx_gzip_without_torchvision(tmp_path):
    from minicnn.data.mnist import load_mnist, load_mnist_test, normalize_mnist

    _write_fake_mnist(tmp_path)
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist(
        tmp_path,
        n_train=2,
        n_val=2,
        seed=0,
        download=False,
    )

    assert x_train.shape == (2, 1, 28, 28)
    assert x_val.shape == (2, 1, 28, 28)
    assert x_test.shape == (3, 1, 28, 28)
    assert y_train.dtype == np.int64
    assert y_val.tolist() == [4, 5]
    assert y_test.tolist() == [4, 5, 6]
    assert normalize_mnist(x_train).dtype == np.float32

    x_only_test, y_only_test = load_mnist_test(tmp_path, download=False)
    assert x_only_test.shape == (3, 1, 28, 28)
    assert y_only_test.tolist() == [4, 5, 6]


def test_flex_mnist_dataloaders_use_local_idx_files(tmp_path):
    from minicnn.flex.data import create_dataloaders, create_test_dataloader

    _write_fake_mnist(tmp_path)
    dataset_cfg = {
        'type': 'mnist',
        'data_root': str(tmp_path),
        'download': False,
        'num_samples': 3,
        'val_samples': 2,
        'input_shape': [1, 28, 28],
        'seed': 123,
    }
    train_cfg = {'batch_size': 2, 'num_workers': 0, 'seed': 456}

    train_loader, val_loader = create_dataloaders(dataset_cfg, train_cfg)
    xb, yb = next(iter(train_loader))
    assert tuple(xb.shape[1:]) == (1, 28, 28)
    assert str(yb.dtype) == 'torch.int64'
    assert len(val_loader.dataset) == 2

    test_loader = create_test_dataloader(dataset_cfg, train_cfg)
    assert test_loader is not None
    assert len(test_loader.dataset) == 3


def test_repository_templates_materialize_supported_torch_models():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / 'templates').glob('*/*.yaml')):
        cfg = load_flex_config(path, [])
        model_cfg = cfg.get('model', {})
        backend = cfg.get('engine', {}).get('backend') or cfg.get('backend', {}).get('type')
        if backend == 'cuda' or 'conv_layers' in model_cfg:
            continue
        model = build_model(model_cfg, cfg['dataset']['input_shape'])
        assert model.inferred_shapes[-1] == (cfg['dataset']['num_classes'],)


def test_refactored_mnist_so_example_keeps_optimizer_separate_from_layers():
    source = (Path(__file__).resolve().parents[1] / 'docs' / 'train_mnist_so_full_cnn_frame.py').read_text()

    assert 'class ConvBlock' in source
    assert 'class DenseLayer' in source
    assert 'class SgdOptimizer' in source
    assert 'class ConvBlockCache' in source
    assert 'def conv2d_out_size' in source
    assert 'def pool2d_out_size' in source

    conv_backward = source.split('class ConvBlock:', 1)[1].split('class DenseLayer:', 1)[0]
    dense_backward = source.split('class DenseLayer:', 1)[1].split('class Mnist2ConvModel:', 1)[0]
    optimizer_body = source.split('class SgdOptimizer:', 1)[1].split('# -----------------------------\n# Layers', 1)[0]

    assert 'apply_sgd_update' not in conv_backward
    assert 'apply_sgd_update' not in dense_backward
    assert 'apply_sgd_update' in optimizer_body
