"""Tests for fixes from code review."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from minicnn.nn.tensor import Parameter, Tensor
from minicnn.optim.sgd import SGD
from minicnn.ops.nn_ops import maxpool2d


# ---------------------------------------------------------------------------
# Fix 1: SGD emits RuntimeWarning instead of silently swallowing exceptions
# ---------------------------------------------------------------------------

def test_sgd_step_warns_on_shape_mismatch():
    w = Parameter([1.0, 2.0], name='w')
    opt = SGD([w], lr=0.1)
    w.grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # wrong shape -> ValueError

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = opt.step()

    assert any(issubclass(c.category, RuntimeWarning) for c in caught), \
        "Expected RuntimeWarning for shape-mismatched grad"
    assert result['updated'] == 0


def test_sgd_step_warns_contains_param_info():
    w = Parameter([1.0, 2.0], name='my_weight')
    opt = SGD([w], lr=0.1)
    w.grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # (3,) vs (2,) -> ValueError

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()

    messages = [str(c.message) for c in caught if issubclass(c.category, RuntimeWarning)]
    assert any('my_weight' in m for m in messages), \
        "Warning should include parameter name"


def test_sgd_step_continues_other_params_after_bad_one():
    bad = Parameter([1.0, 2.0], name='bad')
    good = Parameter([0.0], name='good')
    opt = SGD([bad, good], lr=0.1)
    bad.grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # wrong shape
    good.grad = np.array([1.0], dtype=np.float32)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = opt.step()

    assert result['updated'] == 1
    assert np.allclose(good.data, [-0.1])


def test_sgd_momentum_two_steps_matches_expected_velocity():
    w = Parameter([0.0], name='w')
    opt = SGD([w], lr=0.1, momentum=0.9)

    w.grad = np.array([1.0], dtype=np.float32)
    result1 = opt.step()
    assert result1['updated'] == 1
    assert np.allclose(w.data, [-0.1], atol=1e-6)
    assert np.allclose(opt.velocities[0], [-0.1], atol=1e-6)

    w.grad = np.array([1.0], dtype=np.float32)
    result2 = opt.step()
    assert result2['updated'] == 1
    assert np.allclose(w.data, [-0.29], atol=1e-6)
    assert np.allclose(opt.velocities[0], [-0.19], atol=1e-6)


# ---------------------------------------------------------------------------
# Fix 2: __pow__ backward doesn't produce NaN for zero base with negative power
# ---------------------------------------------------------------------------

def test_pow_negative_exponent_zero_base_no_nan_grad():
    x = Tensor(np.array([0.0, 1.0, 2.0], dtype=np.float32), requires_grad=True)
    y = (x ** -1.0).sum()
    y.backward()

    assert not np.any(np.isnan(x.grad)), \
        "Gradient should not contain NaN for 0**-1"
    assert np.allclose(x.grad[1:], [-1.0, -0.25], atol=1e-6)


def test_pow_negative_exponent_zero_base_grad_is_zero():
    x = Tensor(np.array([0.0], dtype=np.float32), requires_grad=True)
    y = (x ** -2.0).sum()
    y.backward()
    assert x.grad[0] == 0.0


def test_pow_positive_exponent_unchanged():
    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = (x ** 2.0).sum()
    y.backward()
    assert np.allclose(x.grad, [4.0, 6.0])


def test_pow_fractional_exponent():
    x = Tensor(np.array([4.0], dtype=np.float32), requires_grad=True)
    y = (x ** 0.5).sum()
    y.backward()
    assert np.allclose(x.grad, [0.25], atol=1e-6)


# ---------------------------------------------------------------------------
# Fix 3: flex/builder raises ValueError for invalid (non-positive) output shapes
# ---------------------------------------------------------------------------

def test_build_model_raises_for_too_large_kernel():
    from minicnn.flex.builder import build_model

    cfg = {'layers': [
        {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 10},
    ]}
    with pytest.raises(ValueError, match="invalid output shape"):
        build_model(cfg, input_shape=(1, 4, 4))  # 4x4 input, kernel 10 -> negative


def test_build_model_raises_for_bad_maxpool():
    from minicnn.flex.builder import build_model

    cfg = {'layers': [
        {'type': 'MaxPool2d', 'kernel_size': 8},
    ]}
    with pytest.raises(ValueError, match="invalid output shape"):
        build_model(cfg, input_shape=(1, 3, 3))  # 3x3 input, pool 8 -> negative


def test_build_model_valid_shape_succeeds():
    from minicnn.flex.builder import build_model

    cfg = {'layers': [
        {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 3, 'padding': 1},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]}
    model = build_model(cfg, input_shape=(3, 8, 8))
    assert model is not None


# ---------------------------------------------------------------------------
# Fix 5: maxpool2d vectorized forward + backward correctness
# ---------------------------------------------------------------------------

def test_maxpool2d_forward_correctness():
    x_data = np.array([[[[1., 3., 2., 4.],
                         [5., 6., 7., 8.],
                         [9., 2., 1., 3.],
                         [4., 5., 6., 7.]]]], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    out = maxpool2d(x, kernel_size=2, stride=2)

    expected = np.array([[[[6., 8.],
                           [9., 7.]]]], dtype=np.float32)
    assert np.allclose(out.data, expected), f"Forward failed: {out.data}"


def test_maxpool2d_backward_correctness():
    x_data = np.array([[[[1., 3.],
                         [5., 2.]]]], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    out = maxpool2d(x, kernel_size=2, stride=1)
    out.backward(np.ones_like(out.data))

    # Max is at position (1,0) = 5.0
    expected_grad = np.array([[[[0., 0.], [1., 0.]]]], dtype=np.float32)
    assert np.allclose(x.grad, expected_grad), f"Backward failed: {x.grad}"


def test_maxpool2d_backward_scatter_overlap():
    """When multiple outputs map to the same input cell, grads should accumulate."""
    x_data = np.array([[[[9., 1.],
                         [1., 1.]]]], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    out = maxpool2d(x, kernel_size=2, stride=1)
    out.backward(np.ones_like(out.data))
    assert x.grad[0, 0, 0, 0] == 1.0


# ---------------------------------------------------------------------------
# R05: conv2d backward dx now fully vectorized (no kh*kw Python loop)
# ---------------------------------------------------------------------------

def _conv2d_f64(x64, w64, padding=0, stride=1):
    """Pure float64 convolution for use in finite-difference tests."""
    from numpy.lib.stride_tricks import sliding_window_view
    n, c_in, h, w = x64.shape
    c_out, _, kh, kw = w64.shape
    x_pad = np.pad(x64, ((0,0),(0,0),(padding,padding),(padding,padding)))
    windows = sliding_window_view(x_pad, (kh, kw), axis=(2,3))[:,:,::stride,::stride]
    return np.einsum('ncijhw,ochw->noij', windows, w64, optimize=True)


def test_conv2d_backward_dx_correctness():
    """dx from conv2d backward must match float64 finite-difference gradient."""
    rng = np.random.default_rng(7)
    x_data = rng.normal(size=(2, 1, 5, 5)).astype(np.float32)
    w_data = rng.normal(size=(3, 1, 3, 3)).astype(np.float32)

    from minicnn.ops.nn_ops import conv2d

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    out = conv2d(x, w, padding=1)
    out.backward(np.ones_like(out.data))

    x64 = x_data.astype(np.float64)
    w64 = w_data.astype(np.float64)
    eps = 1e-5
    dx_fd = np.zeros_like(x64)
    for idx in np.ndindex(*x64.shape):
        xp, xm = x64.copy(), x64.copy()
        xp[idx] += eps
        xm[idx] -= eps
        dx_fd[idx] = (_conv2d_f64(xp, w64, padding=1).sum() - _conv2d_f64(xm, w64, padding=1).sum()) / (2 * eps)

    assert np.allclose(x.grad, dx_fd, atol=1e-3), \
        f"max error: {np.abs(x.grad - dx_fd).max()}"


def test_conv2d_backward_dw_correctness():
    rng = np.random.default_rng(11)
    x_data = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
    w_data = rng.normal(size=(2, 2, 2, 2)).astype(np.float32)

    from minicnn.ops.nn_ops import conv2d

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    out = conv2d(x, w, stride=1, padding=0)
    out.backward(np.ones_like(out.data))

    eps = 1e-4
    dw_fd = np.zeros_like(w_data)
    for idx in np.ndindex(*w_data.shape):
        wp, wm = w_data.copy(), w_data.copy()
        wp[idx] += eps
        wm[idx] -= eps
        op = conv2d(Tensor(x_data), Tensor(wp)).data.sum()
        om = conv2d(Tensor(x_data), Tensor(wm)).data.sum()
        dw_fd[idx] = (op - om) / (2 * eps)

    assert np.allclose(w.grad, dw_fd, atol=1e-2), \
        f"max error: {np.abs(w.grad - dw_fd).max()}"


def test_maxpool2d_batch_multichannel():
    rng = np.random.default_rng(42)
    x_data = rng.normal(size=(2, 3, 8, 8)).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    out = maxpool2d(x, kernel_size=2, stride=2)

    assert out.data.shape == (2, 3, 4, 4)
    out.backward(np.ones_like(out.data))
    assert x.grad.shape == x_data.shape
    assert not np.any(np.isnan(x.grad))
