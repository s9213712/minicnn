"""Phase 11: Architecture block preset tests."""
from __future__ import annotations
import pytest


def test_conv_relu_expansion():
    from minicnn.flex.builder import _expand_presets
    layers = [{'type': 'conv_relu', 'out_channels': 16, 'kernel_size': 3, 'padding': 1}]
    expanded = _expand_presets(layers)
    assert len(expanded) == 2
    assert expanded[0]['type'] == 'Conv2d'
    assert expanded[0]['out_channels'] == 16
    assert expanded[1]['type'] == 'ReLU'


def test_conv_bn_relu_expansion():
    from minicnn.flex.builder import _expand_presets
    layers = [{'type': 'conv_bn_relu', 'out_channels': 32, 'kernel_size': 3, 'padding': 1}]
    expanded = _expand_presets(layers)
    assert len(expanded) == 3
    assert [e['type'] for e in expanded] == ['Conv2d', 'BatchNorm2d', 'ReLU']
    assert expanded[0]['out_channels'] == 32


def test_conv_bn_silu_expansion():
    from minicnn.flex.builder import _expand_presets
    layers = [{'type': 'conv_bn_silu', 'out_channels': 64, 'kernel_size': 3, 'padding': 1}]
    expanded = _expand_presets(layers)
    assert len(expanded) == 3
    assert [e['type'] for e in expanded] == ['Conv2d', 'BatchNorm2d', 'SiLU']


def test_build_model_with_conv_bn_relu_preset():
    pytest.importorskip('torch')
    import torch
    from minicnn.flex.builder import build_model
    model = build_model(
        {'layers': [
            {'type': 'conv_bn_relu', 'out_channels': 8, 'kernel_size': 3, 'padding': 1},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 4},
        ]},
        input_shape=(3, 8, 8),
    )
    x = torch.randn(2, 3, 8, 8)
    out = model(x)
    assert out.shape == (2, 4)


def test_non_preset_layers_pass_through():
    from minicnn.flex.builder import _expand_presets
    layers = [
        {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 3},
        {'type': 'ReLU'},
    ]
    assert _expand_presets(layers) == layers
