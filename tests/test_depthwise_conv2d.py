from __future__ import annotations


def test_depthwise_conv2d_forward_preserves_spatial_shape():
    import torch

    from minicnn.flex.components import DepthwiseConv2d

    layer = DepthwiseConv2d(in_channels=8, kernel_size=7)
    y = layer(torch.randn(2, 8, 16, 16))

    assert y.shape == (2, 8, 16, 16)
    assert layer.groups == 8
