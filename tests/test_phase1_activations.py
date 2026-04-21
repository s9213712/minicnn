from __future__ import annotations

import numpy as np
import pytest
from minicnn.nn.tensor import Tensor, Parameter
from minicnn.nn.layers import LeakyReLU, SiLU


def test_leaky_relu_forward_positive():
    x = Tensor(np.array([1.0, 2.0, -1.0, -3.0], dtype=np.float32))
    layer = LeakyReLU(negative_slope=0.1)
    out = layer(x)
    expected = np.array([1.0, 2.0, -0.1, -0.3], dtype=np.float32)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


def test_leaky_relu_backward():
    x = Parameter(np.array([2.0, -2.0], dtype=np.float32))
    layer = LeakyReLU(negative_slope=0.2)
    out = layer(x)
    out.backward(np.ones_like(out.data))
    expected = np.array([1.0, 0.2], dtype=np.float32)
    np.testing.assert_allclose(x.grad, expected, rtol=1e-6)


def test_leaky_relu_default_slope():
    layer = LeakyReLU()
    assert layer.negative_slope == 0.01
    x = Tensor(np.array([-1.0], dtype=np.float32))
    out = layer(x)
    np.testing.assert_allclose(out.data, [-0.01], rtol=1e-6)


def test_silu_forward():
    x_val = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    x = Tensor(x_val)
    layer = SiLU()
    out = layer(x)
    s = 1.0 / (1.0 + np.exp(-x_val))
    expected = (x_val * s).astype(np.float32)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


def test_silu_backward():
    x_val = np.array([1.0, -1.0], dtype=np.float32)
    x = Parameter(x_val.copy())
    layer = SiLU()
    out = layer(x)
    out.backward(np.ones_like(out.data))
    s = 1.0 / (1.0 + np.exp(-x_val))
    expected = (s + x_val * s * (1.0 - s)).astype(np.float32)
    np.testing.assert_allclose(x.grad, expected, rtol=1e-5)


def test_builder_leaky_relu_config():
    from minicnn.models.builder import build_model_from_config
    cfg = {
        'input_shape': [4],
        'layers': [
            {'type': 'Linear', 'out_features': 8},
            {'type': 'LeakyReLU', 'negative_slope': 0.05},
            {'type': 'Linear', 'out_features': 2},
        ],
    }
    model, _ = build_model_from_config(cfg)
    import minicnn.nn.layers as L
    leaky = model[1]
    assert isinstance(leaky, L.LeakyReLU)
    assert leaky.negative_slope == 0.05


def test_builder_silu_config():
    from minicnn.models.builder import build_model_from_config
    cfg = {
        'input_shape': [4],
        'layers': [
            {'type': 'Linear', 'out_features': 8},
            {'type': 'SiLU'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }
    model, _ = build_model_from_config(cfg)
    import minicnn.nn.layers as L
    silu = model[1]
    assert isinstance(silu, L.SiLU)


def test_flex_sigmoid_registered():
    from minicnn.flex.registry import REGISTRY
    from minicnn.flex import components  # noqa: F401
    assert REGISTRY.has('activations', 'Sigmoid')
