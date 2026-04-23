from __future__ import annotations

import warnings


def _base_cfg(tmp_path):
    return {
        'engine': {'backend': 'cuda_native'},
        'dataset': {
            'type': 'random',
            'input_shape': [3, 32, 32],
            'num_classes': 10,
            'num_samples': 16,
            'val_samples': 8,
        },
        'train': {'batch_size': 8, 'epochs': 1, 'device': 'cpu', 'amp': False, 'grad_accum_steps': 1},
        'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9},
        'loss': {'type': 'CrossEntropyLoss'},
        'scheduler': {'enabled': False},
        'run': {'output_root': str(tmp_path)},
    }


def test_validate_cuda_native_accepts_residual_block(tmp_path):
    from minicnn.cuda_native.api import validate_cuda_native_config

    cfg = _base_cfg(tmp_path)
    cfg['model'] = {
        'layers': [
            {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ResidualBlock', 'channels': 16, 'stride': 1},
            {'type': 'GlobalAvgPool2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ],
    }

    assert validate_cuda_native_config(cfg) == []


def test_validate_cuda_native_accepts_dropout(tmp_path):
    from minicnn.cuda_native.api import validate_cuda_native_config

    cfg = _base_cfg(tmp_path)
    cfg['model'] = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 32},
            {'type': 'Dropout', 'p': 0.25},
            {'type': 'Linear', 'out_features': 10},
        ],
    }

    assert validate_cuda_native_config(cfg) == []


def test_validate_cuda_native_accepts_named_convnext_tiny(tmp_path):
    from minicnn.cuda_native.api import validate_cuda_native_config

    cfg = _base_cfg(tmp_path)
    cfg['model'] = {'name': 'convnext_tiny', 'stem_channels': 16, 'stage2_channels': 32}

    assert validate_cuda_native_config(cfg) == []


def test_build_cuda_native_graph_resolves_named_convnext_tiny():
    from minicnn.cuda_native.api import build_cuda_native_graph

    graph = build_cuda_native_graph(
        {'name': 'convnext_tiny', 'stem_channels': 16, 'stage2_channels': 32},
        (8, 3, 32, 32),
    )

    assert graph.nodes
    assert graph.nodes[0].op_type == 'Conv2d'
    assert any(node.op_type == 'ConvNeXtBlock' for node in graph.nodes)


def test_trainer_accepts_residual_block_on_cuda_native(tmp_path):
    from minicnn.unified.trainer import train_unified_from_config

    cfg = _base_cfg(tmp_path)
    cfg['model'] = {
        'layers': [
            {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ResidualBlock', 'channels': 16, 'stride': 1},
            {'type': 'GlobalAvgPool2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_dir = train_unified_from_config(cfg)

    assert run_dir.exists()
    assert (run_dir / 'summary.json').exists()


def test_trainer_accepts_convnext_block_on_cuda_native(tmp_path):
    from minicnn.unified.trainer import train_unified_from_config

    cfg = _base_cfg(tmp_path)
    cfg['model'] = {
        'layers': [
            {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ConvNeXtBlock'},
            {'type': 'GlobalAvgPool2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_dir = train_unified_from_config(cfg)

    assert run_dir.exists()
    assert (run_dir / 'summary.json').exists()


def test_trainer_accepts_dropout_on_cuda_native(tmp_path):
    from minicnn.unified.trainer import train_unified_from_config

    cfg = _base_cfg(tmp_path)
    cfg['model'] = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 32},
            {'type': 'Dropout', 'p': 0.25},
            {'type': 'Linear', 'out_features': 10},
        ],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_dir = train_unified_from_config(cfg)

    assert run_dir.exists()
    assert (run_dir / 'summary.json').exists()
