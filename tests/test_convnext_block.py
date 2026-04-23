from __future__ import annotations


def test_convnext_block_forward_preserves_residual_shape():
    import torch

    from minicnn.flex.components import ConvNeXtBlock

    block = ConvNeXtBlock(channels=16)
    y = block(torch.randn(2, 16, 8, 8))

    assert y.shape == (2, 16, 8, 8)
