from __future__ import annotations


def test_layernorm2d_preserves_nchw_shape():
    import torch

    from minicnn.flex.components import LayerNorm2d

    layer = LayerNorm2d(4)
    x = torch.randn(2, 4, 8, 8)
    y = layer(x)

    assert y.shape == x.shape
    assert tuple(layer.weight.shape) == (4,)
    assert tuple(layer.bias.shape) == (4,)


def test_explicit_convnext_primitives_build_and_run_forward():
    import torch

    from minicnn.flex.builder import build_model

    model = build_model({
        'layers': [
            {'type': 'DepthwiseConv2d', 'kernel_size': 7},
            {'type': 'LayerNorm2d'},
            {'type': 'PointwiseConv2d', 'out_channels': 16},
            {'type': 'GELU'},
            {'type': 'PointwiseConv2d', 'out_channels': 8},
        ],
    }, (8, 16, 16))

    y = model(torch.randn(2, 8, 16, 16))

    assert y.shape == (2, 8, 16, 16)


def test_convnext_primitives_are_registered():
    from minicnn.flex.registry import describe_registries

    summary = describe_registries()

    assert 'DepthwiseConv2d' in summary['layers']
    assert 'depthwise_conv2d' in summary['layers']
    assert 'PointwiseConv2d' in summary['layers']
    assert 'pointwise_conv2d' in summary['layers']
    assert 'LayerNorm2d' in summary['layers']
    assert 'layernorm2d' in summary['layers']
    assert 'ConvNeXtBlock' in summary['layers']
    assert 'convnext_block' in summary['layers']
