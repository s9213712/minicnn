from __future__ import annotations


def test_pointwise_conv2d_changes_only_channel_dimension():
    import torch

    from minicnn.flex.components import PointwiseConv2d

    layer = PointwiseConv2d(in_channels=8, out_channels=16)
    y = layer(torch.randn(2, 8, 16, 16))

    assert y.shape == (2, 16, 16, 16)
    assert layer.kernel_size == (1, 1)
