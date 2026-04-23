from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _write_fake_cifar_batch(path: Path, *, use_bytes_keys: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_key = b'data' if use_bytes_keys else 'data'
    labels_key = b'labels' if use_bytes_keys else 'labels'
    payload = {
        data_key: np.arange(2 * 3 * 32 * 32, dtype=np.uint8).reshape(2, -1),
        labels_key: [1, 7],
    }
    with path.open('wb') as fh:
        pickle.dump(payload, fh, protocol=2)


def test_load_batch_reads_string_keys_from_python2_style_pickle(tmp_path):
    from minicnn.data.cifar10 import _load_batch

    batch_path = tmp_path / 'data_batch_1'
    _write_fake_cifar_batch(batch_path, use_bytes_keys=False)

    x, y = _load_batch(batch_path)

    assert x.shape == (2, 3, 32, 32)
    assert x.dtype == np.float32
    assert y.tolist() == [1, 7]


def test_load_batch_reads_bytes_keys_and_uses_latin1(monkeypatch, tmp_path):
    from minicnn.data.cifar10 import _load_batch
    import minicnn.data.cifar10 as cifar10

    batch_path = tmp_path / 'data_batch_1'
    _write_fake_cifar_batch(batch_path, use_bytes_keys=True)

    real_pickle_load = cifar10.pickle.load
    calls: list[str] = []

    def _spy_pickle_load(file_obj, *, encoding):
        calls.append(encoding)
        return real_pickle_load(file_obj, encoding=encoding)

    monkeypatch.setattr(cifar10.pickle, 'load', _spy_pickle_load)

    x, y = _load_batch(batch_path)

    assert calls == ['latin1']
    assert x.shape == (2, 3, 32, 32)
    assert y.tolist() == [1, 7]
