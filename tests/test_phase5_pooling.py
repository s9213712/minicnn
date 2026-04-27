from __future__ import annotations

import numpy as np
import pytest
from minicnn.nn.layers import AvgPool2d
from minicnn.nn.tensor import Parameter, Tensor


def test_avgpool2d_forward_values():
    x_data = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)
    x = Tensor(x_data)
    layer = AvgPool2d(kernel_size=2, stride=2)
    out = layer(x)
    assert out.data.shape == (1, 1, 2, 2)
    expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]], dtype=np.float32)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


def test_avgpool2d_forward_single_window():
    x_data = np.ones((1, 1, 2, 2), dtype=np.float32) * 4.0
    x = Tensor(x_data)
    layer = AvgPool2d(kernel_size=2, stride=2)
    out = layer(x)
    np.testing.assert_allclose(out.data, np.array([[[[4.0]]]], dtype=np.float32), rtol=1e-6)


def test_avgpool2d_backward_sum():
    x_data = np.ones((1, 1, 4, 4), dtype=np.float32)
    x = Parameter(x_data)
    layer = AvgPool2d(kernel_size=2, stride=2)
    out = layer(x)
    out.backward(np.ones_like(out.data))
    assert x.grad is not None
    assert x.grad.shape == x.data.shape
    np.testing.assert_allclose(x.grad.sum(), out.data.size, rtol=1e-5)


def test_avgpool2d_backward_gradient_per_window():
    x_data = np.ones((1, 1, 2, 2), dtype=np.float32)
    x = Parameter(x_data)
    layer = AvgPool2d(kernel_size=2, stride=2)
    out = layer(x)
    out.backward(np.ones_like(out.data))
    np.testing.assert_allclose(x.grad, np.full((1, 1, 2, 2), 0.25, dtype=np.float32), rtol=1e-6)


def test_avgpool2d_backward_overlapping_windows_accumulates_correctly():
    x_data = np.ones((1, 1, 3, 3), dtype=np.float32)
    x = Parameter(x_data)
    layer = AvgPool2d(kernel_size=2, stride=1)
    out = layer(x)
    out.backward(np.ones_like(out.data))
    expected = np.array(
        [[[[0.25, 0.5, 0.25],
           [0.5, 1.0, 0.5],
           [0.25, 0.5, 0.25]]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(x.grad, expected, rtol=1e-6)


def test_avgpool2d_padding_matches_shape_and_zero_padding_average():
    x_data = np.ones((1, 1, 2, 2), dtype=np.float32)
    x = Tensor(x_data)
    layer = AvgPool2d(kernel_size=2, stride=2, padding=1)
    out = layer(x)
    expected = np.full((1, 1, 2, 2), 0.25, dtype=np.float32)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


def test_avgpool2d_builder():
    from minicnn.models.builder import build_model_from_config
    cfg = {
        'input_shape': [1, 4, 4],
        'layers': [
            {'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }
    model, final_shape = build_model_from_config(cfg)
    assert final_shape == (2,)


def test_pool_shape_inference_respects_padding():
    from minicnn.models.builder import build_model_from_config

    cfg = {
        'input_shape': [1, 4, 4],
        'layers': [
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 1},
            {'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 1, 'padding': 1},
            {'type': 'Flatten'},
        ],
    }
    _model, final_shape = build_model_from_config(cfg)
    assert final_shape == (16,)
