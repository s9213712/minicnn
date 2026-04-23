from __future__ import annotations


def test_layernorm2d_runs_on_nchw_input():
    import torch

    from minicnn.flex.components import LayerNorm2d

    layer = LayerNorm2d(4)
    y = layer(torch.randn(2, 4, 8, 8))

    assert y.shape == (2, 4, 8, 8)
