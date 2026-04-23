from __future__ import annotations

import warnings


def _minimal_convnext_native_cfg(tmp_path):
    return {
        'engine': {'backend': 'cuda_native'},
        'dataset': {
            'type': 'random',
            'input_shape': [3, 32, 32],
            'num_classes': 10,
            'num_samples': 16,
            'val_samples': 8,
        },
        'model': {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'type': 'DepthwiseConv2d', 'kernel_size': 7},
                {'type': 'LayerNorm2d'},
                {'type': 'PointwiseConv2d', 'out_channels': 64},
                {'type': 'GELU'},
                {'type': 'PointwiseConv2d', 'out_channels': 16},
                {'type': 'GlobalAvgPool2d'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 10},
            ],
        },
        'train': {'batch_size': 8, 'epochs': 1, 'device': 'cpu', 'amp': False, 'grad_accum_steps': 1},
        'optimizer': {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9},
        'loss': {'type': 'CrossEntropyLoss'},
        'scheduler': {'enabled': False},
        'run': {'output_root': str(tmp_path)},
    }


def test_cuda_native_capabilities_include_convnext_primitives():
    from minicnn.cuda_native.api import get_capability_summary

    caps = get_capability_summary()

    assert caps['supports_depthwise_conv'] is True
    assert caps['supports_pointwise_conv'] is True
    assert caps['supports_layernorm2d'] is True
    assert caps['supports_gelu'] is True
    assert caps['supports_residual_add'] is True
    assert caps['supports_convnext_block'] is True
    for op in (
        'DepthwiseConv2d',
        'PointwiseConv2d',
        'LayerNorm2d',
        'GELU',
        'GlobalAvgPool2d',
        'AdaptiveAvgPool2d',
        'Identity',
        'ResidualBlock',
        'ConvNeXtBlock',
        'Dropout',
    ):
        assert op in caps['supported_ops']


def test_validate_cuda_native_accepts_explicit_convnext_primitives(tmp_path):
    from minicnn.cuda_native.api import validate_cuda_native_config

    errors = validate_cuda_native_config(_minimal_convnext_native_cfg(tmp_path))
    assert errors == []


def test_trainer_accepts_explicit_convnext_primitives_on_cuda_native(tmp_path):
    from minicnn.unified.trainer import train_unified_from_config

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_dir = train_unified_from_config(_minimal_convnext_native_cfg(tmp_path))

    assert run_dir.exists()
    assert (run_dir / 'summary.json').exists()
