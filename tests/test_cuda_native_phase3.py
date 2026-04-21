"""Phase 3 tests: backward kernels, loss functions, and minimal training loop."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def test_cross_entropy_loss_value():
    from minicnn.cuda_native.loss import cross_entropy_loss
    logits = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
    labels = np.array([0])
    loss, grad = cross_entropy_loss(logits, labels)
    assert 0.0 < loss < 2.0
    assert grad.shape == logits.shape


def test_cross_entropy_grad_shape():
    from minicnn.cuda_native.loss import cross_entropy_loss
    logits = np.random.randn(8, 10).astype(np.float32)
    labels = np.random.randint(0, 10, size=8)
    loss, grad = cross_entropy_loss(logits, labels)
    assert grad.shape == (8, 10)
    assert np.isfinite(loss)


def test_cross_entropy_grad_sums_to_zero():
    """For a batch of 1 the gradient rows should sum to 0."""
    from minicnn.cuda_native.loss import cross_entropy_loss
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    labels = np.array([2])
    _, grad = cross_entropy_loss(logits, labels)
    np.testing.assert_allclose(grad.sum(), 0.0, atol=1e-6)


def test_mse_loss_value():
    from minicnn.cuda_native.loss import mse_loss
    preds = np.array([[1.0, 2.0]], dtype=np.float32)
    targets = np.array([[1.0, 2.0]], dtype=np.float32)
    loss, grad = mse_loss(preds, targets)
    assert loss == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_equal(grad, np.zeros_like(grad))


def test_mse_loss_grad():
    from minicnn.cuda_native.loss import mse_loss
    preds = np.array([[3.0]], dtype=np.float32)
    targets = np.array([[1.0]], dtype=np.float32)
    loss, grad = mse_loss(preds, targets)
    assert loss == pytest.approx(4.0, rel=1e-5)
    assert grad[0, 0] == pytest.approx(4.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Forward with cache
# ---------------------------------------------------------------------------

def test_run_with_cache_saves_inputs():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 3},
    ], (2, 4))
    x = np.random.randn(2, 4).astype(np.float32)
    w = np.random.randn(3, 4).astype(np.float32)
    ctx, cache = ForwardExecutor().run_with_cache(
        g, {'input': x}, params={'_w_linear_2': w}
    )
    assert 'fwd_relu_0_in' in cache
    assert 'fwd_flatten_1_in_shape' in cache
    assert 'fwd_linear_2_in' in cache
    assert '_w_linear_2' in cache


# ---------------------------------------------------------------------------
# Individual backward kernels
# ---------------------------------------------------------------------------

def test_bwd_relu_positive():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'ReLU'}], (1, 4))
    x = np.array([[2.0, -1.0, 0.5, -3.0]], dtype=np.float32)
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x})
    grad_out = np.ones((1, 4), dtype=np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)
    expected = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(grad_in, expected)


def test_bwd_leaky_relu():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'LeakyReLU', 'negative_slope': 0.1}], (1, 2))
    x = np.array([[1.0, -2.0]], dtype=np.float32)
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x})
    grad_out = np.ones((1, 2), dtype=np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)
    np.testing.assert_allclose(grad_in, [[1.0, 0.1]], rtol=1e-5)


@pytest.mark.parametrize(
    ('op_type', 'x', 'expected'),
    [
        ('Sigmoid', np.array([[0.0]], dtype=np.float32), np.array([[0.25]], dtype=np.float32)),
        ('Tanh', np.array([[0.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32)),
        ('SiLU', np.array([[0.0]], dtype=np.float32), np.array([[0.5]], dtype=np.float32)),
    ],
)
def test_bwd_extra_activations(op_type, x, expected):
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': op_type}], (1, 1))
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x})
    grad_out = np.ones((1, 1), dtype=np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)
    np.testing.assert_allclose(grad_in, expected, atol=1e-6)


def test_bwd_flatten_restores_shape():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'Flatten'}], (2, 3, 4, 4))
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x})
    grad_out = np.ones((2, 48), dtype=np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)
    assert grad_in.shape == (2, 3, 4, 4)


def test_bwd_linear_param_grads_shape():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'Linear', 'out_features': 3}], (4, 5))
    x = np.random.randn(4, 5).astype(np.float32)
    w = np.random.randn(3, 5).astype(np.float32)
    b = np.zeros(3, dtype=np.float32)
    params = {'_w_linear_0': w, '_b_linear_0': b}
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x}, params=params)
    grad_out = np.ones((4, 3), dtype=np.float32)
    grad_in, param_grads = BackwardExecutor().run(g, grad_out, cache)
    assert grad_in.shape == (4, 5)
    assert param_grads['_w_linear_0'].shape == (3, 5)
    assert param_grads['_b_linear_0'].shape == (3,)


def test_bwd_linear_grad_correctness():
    """Numerical gradient check for Linear backward."""
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor

    np.random.seed(42)
    g = build_graph([{'type': 'Linear', 'out_features': 2}], (3, 4))
    x = np.random.randn(3, 4).astype(np.float32)
    w = np.random.randn(2, 4).astype(np.float32)
    params = {'_w_linear_0': w}

    # Analytic gradient
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x}, params=params)
    grad_out = np.ones((3, 2), dtype=np.float32)
    grad_in, pg = BackwardExecutor().run(g, grad_out, cache)

    # Numerical gradient for w[0,0]
    eps = 1e-4
    w_plus = w.copy(); w_plus[0, 0] += eps
    w_minus = w.copy(); w_minus[0, 0] -= eps
    fwd = ForwardExecutor()
    out_plus = fwd.run_inference(g, x, {'_w_linear_0': w_plus})
    out_minus = fwd.run_inference(g, x, {'_w_linear_0': w_minus})
    # loss = sum(out), so dL/dw[0,0] = (sum(out+) - sum(out-))/2eps
    num_grad = float((out_plus.sum() - out_minus.sum()) / (2 * eps))
    np.testing.assert_allclose(pg['_w_linear_0'][0, 0], num_grad, rtol=2e-2)


def test_bwd_conv2d_grad_shapes():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3}], (1, 2, 5, 5))
    x = np.random.randn(1, 2, 5, 5).astype(np.float32)
    w = np.random.randn(4, 2, 3, 3).astype(np.float32)
    params = {'_w_conv2d_0': w}
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x}, params=params)
    grad_out = np.ones((1, 4, 3, 3), dtype=np.float32)
    grad_in, pg = BackwardExecutor().run(g, grad_out, cache)
    assert grad_in.shape == (1, 2, 5, 5)
    assert pg['_w_conv2d_0'].shape == (4, 2, 3, 3)


def test_bwd_conv2d_numerical_grad():
    """Numerical gradient check for Conv2d dL/dW[0,0,0,0]."""
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor

    np.random.seed(7)
    g = build_graph([{'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3}], (1, 1, 5, 5))
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)
    w = np.random.randn(2, 1, 3, 3).astype(np.float32)
    params = {'_w_conv2d_0': w}

    fwd = ForwardExecutor()
    _, cache = fwd.run_with_cache(g, {'input': x}, params=params)
    grad_out = np.ones((1, 2, 3, 3), dtype=np.float32)
    _, pg = BackwardExecutor().run(g, grad_out, cache)

    eps = 1e-4
    w_p = w.copy(); w_p[0, 0, 0, 0] += eps
    w_m = w.copy(); w_m[0, 0, 0, 0] -= eps
    num = float((fwd.run_inference(g, x, {'_w_conv2d_0': w_p}).sum()
                 - fwd.run_inference(g, x, {'_w_conv2d_0': w_m}).sum()) / (2 * eps))
    np.testing.assert_allclose(pg['_w_conv2d_0'][0, 0, 0, 0], num, rtol=1e-2)


def test_bwd_maxpool2d_grad_shape():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2}], (1, 2, 4, 4))
    x = np.random.randn(1, 2, 4, 4).astype(np.float32)
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x})
    grad_out = np.ones((1, 2, 2, 2), dtype=np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)
    assert grad_in.shape == (1, 2, 4, 4)


def test_bwd_avgpool2d_grad_shape():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    g = build_graph([{'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 2}], (1, 2, 4, 4))
    x = np.random.randn(1, 2, 4, 4).astype(np.float32)
    _, cache = ForwardExecutor().run_with_cache(g, {'input': x})
    grad_out = np.ones((1, 2, 2, 2), dtype=np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)
    assert grad_in.shape == (1, 2, 4, 4)


def test_backward_registry_includes_batchnorm2d():
    from minicnn.cuda_native.backward import make_default_backward_registry
    reg = make_default_backward_registry()
    assert reg.has('BatchNorm2d') is True


def test_bwd_batchnorm2d_train_param_grad_shapes():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor

    g = build_graph([{'type': 'BatchNorm2d'}], (2, 3, 4, 4))
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    params = {
        '_w_batchnorm2d_0': np.ones(3, dtype=np.float32),
        '_b_batchnorm2d_0': np.zeros(3, dtype=np.float32),
        '_running_mean_batchnorm2d_0': np.zeros(3, dtype=np.float32),
        '_running_var_batchnorm2d_0': np.ones(3, dtype=np.float32),
    }

    _, cache = ForwardExecutor().run_with_cache(g, {'input': x}, params=params, mode='train')
    grad_out = np.random.randn(2, 3, 4, 4).astype(np.float32)
    grad_in, param_grads = BackwardExecutor().run(g, grad_out, cache)

    assert grad_in.shape == x.shape
    assert param_grads['_w_batchnorm2d_0'].shape == (3,)
    assert param_grads['_b_batchnorm2d_0'].shape == (3,)


def test_bwd_batchnorm2d_train_input_grad_matches_numeric():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor

    np.random.seed(11)
    g = build_graph([{'type': 'BatchNorm2d', 'eps': 1e-5}], (2, 2, 2, 2))
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    params = {
        '_w_batchnorm2d_0': np.array([1.25, 0.75], dtype=np.float32),
        '_b_batchnorm2d_0': np.array([0.1, -0.2], dtype=np.float32),
        '_running_mean_batchnorm2d_0': np.zeros(2, dtype=np.float32),
        '_running_var_batchnorm2d_0': np.ones(2, dtype=np.float32),
    }

    fwd = ForwardExecutor()
    _, cache = fwd.run_with_cache(g, {'input': x}, params=params, mode='train')
    grad_out = np.random.randn(2, 2, 2, 2).astype(np.float32)
    grad_in, _ = BackwardExecutor().run(g, grad_out, cache)

    eps = 1e-3
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[0, 0, 0, 0] += eps
    x_minus[0, 0, 0, 0] -= eps

    out_plus = fwd.run_inference(g, x_plus, params={
        '_w_batchnorm2d_0': params['_w_batchnorm2d_0'].copy(),
        '_b_batchnorm2d_0': params['_b_batchnorm2d_0'].copy(),
        '_running_mean_batchnorm2d_0': np.zeros(2, dtype=np.float32),
        '_running_var_batchnorm2d_0': np.ones(2, dtype=np.float32),
    }, mode='train')
    out_minus = fwd.run_inference(g, x_minus, params={
        '_w_batchnorm2d_0': params['_w_batchnorm2d_0'].copy(),
        '_b_batchnorm2d_0': params['_b_batchnorm2d_0'].copy(),
        '_running_mean_batchnorm2d_0': np.zeros(2, dtype=np.float32),
        '_running_var_batchnorm2d_0': np.ones(2, dtype=np.float32),
    }, mode='train')
    num_grad = float(((out_plus - out_minus) * grad_out).sum() / (2 * eps))
    np.testing.assert_allclose(grad_in[0, 0, 0, 0], num_grad, rtol=5e-2, atol=5e-2)


def test_train_step_with_batchnorm2d_updates_affine_params():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.training import train_step

    np.random.seed(17)
    g = build_graph([
        {'type': 'BatchNorm2d'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 2},
    ], (4, 1, 2, 2))
    x = np.random.randn(4, 1, 2, 2).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    params = {
        '_w_batchnorm2d_0': np.ones(1, dtype=np.float32),
        '_b_batchnorm2d_0': np.zeros(1, dtype=np.float32),
        '_running_mean_batchnorm2d_0': np.zeros(1, dtype=np.float32),
        '_running_var_batchnorm2d_0': np.ones(1, dtype=np.float32),
        '_w_linear_2': np.random.randn(2, 4).astype(np.float32) * 0.1,
        '_b_linear_2': np.zeros(2, dtype=np.float32),
    }

    loss, new_params = train_step(g, x, y, params, lr=0.05)
    assert np.isfinite(loss)
    assert not np.allclose(new_params['_w_batchnorm2d_0'], params['_w_batchnorm2d_0'])
    assert not np.allclose(new_params['_b_batchnorm2d_0'], params['_b_batchnorm2d_0'])


# ---------------------------------------------------------------------------
# Single-step training: params update
# ---------------------------------------------------------------------------

def test_train_step_linear_updates_params():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.training import train_step

    np.random.seed(0)
    g = build_graph([{'type': 'Linear', 'out_features': 3}], (4, 8))
    x = np.random.randn(4, 8).astype(np.float32)
    y = np.array([0, 1, 2, 0])
    w = np.random.randn(3, 8).astype(np.float32)
    params = {'_w_linear_0': w}

    loss, new_params = train_step(g, x, y, params, lr=0.1)
    assert np.isfinite(loss)
    # weights must have changed
    assert not np.allclose(new_params['_w_linear_0'], params['_w_linear_0'])


def test_train_step_loss_decreases():
    """Multiple SGD steps on a tiny problem should reduce loss."""
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.training import train_step

    np.random.seed(1)
    g = build_graph([
        {'type': 'Linear', 'out_features': 4},
        {'type': 'ReLU'},
        {'type': 'Linear', 'out_features': 2},
    ], (8, 6))
    x = np.random.randn(8, 6).astype(np.float32)
    y = np.random.randint(0, 2, size=8)
    params = {
        '_w_linear_0': np.random.randn(4, 6).astype(np.float32) * 0.1,
        '_b_linear_0': np.zeros(4, dtype=np.float32),
        '_w_linear_2': np.random.randn(2, 4).astype(np.float32) * 0.1,
        '_b_linear_2': np.zeros(2, dtype=np.float32),
    }

    first_loss, params = train_step(g, x, y, params, lr=0.05)
    for _ in range(49):
        _, params = train_step(g, x, y, params, lr=0.05)
    last_loss, _ = train_step(g, x, y, params, lr=0.05)
    assert last_loss < first_loss, f'Expected loss decrease: {first_loss:.4f} -> {last_loss:.4f}'


def test_train_step_mse():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.training import train_step

    np.random.seed(2)
    g = build_graph([{'type': 'Linear', 'out_features': 1}], (4, 3))
    x = np.random.randn(4, 3).astype(np.float32)
    y = np.zeros((4, 1), dtype=np.float32)
    w = np.random.randn(1, 3).astype(np.float32)
    params = {'_w_linear_0': w}
    loss, new_params = train_step(g, x, y, params, lr=0.01, loss_type='mse')
    assert np.isfinite(loss)
    assert not np.allclose(new_params['_w_linear_0'], params['_w_linear_0'])


def test_train_step_unknown_loss_raises():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.training import train_step
    g = build_graph([{'type': 'Linear', 'out_features': 2}], (2, 3))
    x = np.zeros((2, 3), dtype=np.float32)
    y = np.array([0, 1])
    params = {'_w_linear_0': np.zeros((2, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match='Unknown loss_type'):
        train_step(g, x, y, params, lr=0.01, loss_type='bce')  # type: ignore


def test_train_step_grad_shapes_match_params():
    """Gradient shapes must exactly match parameter shapes."""
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.loss import cross_entropy_loss

    np.random.seed(3)
    g = build_graph([
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 5},
    ], (2, 3, 4, 4))
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    y = np.array([0, 4])
    w = np.random.randn(5, 48).astype(np.float32)
    params = {'_w_linear_1': w}

    fwd = ForwardExecutor()
    ctx, cache = fwd.run_with_cache(g, {'input': x}, params=params)
    logits = ctx[g.output_spec.name]
    _, grad_logits = cross_entropy_loss(logits, y)
    _, pg = BackwardExecutor().run(g, grad_logits, cache)

    assert pg['_w_linear_1'].shape == w.shape


def test_sgd_update():
    from minicnn.cuda_native.training import sgd_update
    params = {'w': np.array([1.0, 2.0], dtype=np.float32)}
    grads = {'w': np.array([0.1, 0.2], dtype=np.float32)}
    updated = sgd_update(params, grads, lr=1.0)
    np.testing.assert_allclose(updated['w'], [0.9, 1.8], rtol=1e-5)


def test_sgd_update_weight_decay():
    from minicnn.cuda_native.training import sgd_update
    params = {'w': np.array([1.0], dtype=np.float32)}
    grads = {'w': np.array([0.0], dtype=np.float32)}
    updated = sgd_update(params, grads, lr=1.0, weight_decay=0.1)
    # grad_effective = 0 + 0.1*1.0 = 0.1; updated = 1.0 - 1.0*0.1 = 0.9
    np.testing.assert_allclose(updated['w'], [0.9], rtol=1e-5)


def test_sgd_update_momentum():
    from minicnn.cuda_native.training import sgd_update
    params = {'w': np.array([1.0], dtype=np.float32)}
    grads = {'w': np.array([0.5], dtype=np.float32)}
    state: dict[str, dict[str, np.ndarray]] = {}
    updated1 = sgd_update(params, grads, lr=0.1, momentum=0.9, optimizer_state=state)
    updated2 = sgd_update(updated1, grads, lr=0.1, momentum=0.9, optimizer_state=state)
    assert updated1['w'][0] < params['w'][0]
    assert updated2['w'][0] < updated1['w'][0]


def test_sgd_update_gradient_clipping():
    from minicnn.cuda_native.training import sgd_update
    params = {'w': np.array([0.0, 0.0], dtype=np.float32)}
    grads = {'w': np.array([3.0, 4.0], dtype=np.float32)}
    updated = sgd_update(params, grads, lr=1.0, grad_clip_global=1.0)
    np.testing.assert_allclose(updated['w'], np.array([-0.6, -0.8], dtype=np.float32), atol=1e-6)
