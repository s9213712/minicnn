from __future__ import annotations

import numpy as np

from minicnn.nn import Linear, Tensor, set_global_seed
from minicnn.ops.nn_ops import dropout
from minicnn.training._autograd_reporting import resolve_autograd_artifacts
from minicnn.training.train_autograd import train_autograd_from_config


def test_set_global_seed_makes_manual_layer_init_reproducible():
    set_global_seed(123)
    first = Linear(3, 2)
    set_global_seed(123)
    second = Linear(3, 2)

    assert np.array_equal(first.weight.data, second.weight.data)
    assert np.array_equal(first.bias.data, second.bias.data)


def test_set_global_seed_makes_dropout_reproducible():
    x = Tensor(np.arange(8, dtype=np.float32).reshape(2, 4))

    set_global_seed(7)
    y1 = dropout(x, p=0.5, training=True)
    set_global_seed(7)
    y2 = dropout(x, p=0.5, training=True)

    assert np.array_equal(y1.data, y2.data)


def test_train_autograd_dropout_is_reproducible_with_distinct_init_and_train_seeds(tmp_path):
    cfg = {
        'project': {'name': 'test', 'run_name': 'dropout-seeded', 'artifacts_root': str(tmp_path / 'run1')},
        'dataset': {
            'type': 'random',
            'input_shape': [4],
            'num_classes': 2,
            'num_samples': 8,
            'val_samples': 4,
            'test_samples': 4,
            'seed': 101,
        },
        'model': {'layers': [
            {'type': 'Dropout', 'p': 0.5},
            {'type': 'Linear', 'out_features': 2},
        ]},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'loss': {'type': 'CrossEntropyLoss'},
        'train': {'epochs': 2, 'batch_size': 2, 'init_seed': 202, 'train_seed': 303},
    }

    run1 = train_autograd_from_config(cfg)
    cfg['project'] = dict(cfg['project'], run_name='dropout-seeded-2', artifacts_root=str(tmp_path / 'run2'))
    run2 = train_autograd_from_config(cfg)

    ckpt1 = np.load(resolve_autograd_artifacts(run1)[1])
    ckpt2 = np.load(resolve_autograd_artifacts(run2)[1])

    assert ckpt1.files == ckpt2.files
    for key in ckpt1.files:
        assert np.allclose(ckpt1[key], ckpt2[key], atol=1e-6), key
