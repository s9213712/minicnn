"""BatchNorm2d eval-mode reference tests for cuda_native."""
from __future__ import annotations

import numpy as np
import pytest


def test_batchnorm2d_eval_forward_matches_numpy_reference():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    graph = build_graph([{'type': 'BatchNorm2d', 'eps': 1e-5}], (1, 2, 2, 2))
    x = np.array(
        [[[[1.0, 2.0],
           [3.0, 4.0]],
          [[2.0, 4.0],
           [6.0, 8.0]]]],
        dtype=np.float32,
    )
    gamma = np.array([1.5, 0.5], dtype=np.float32)
    beta = np.array([0.25, -0.75], dtype=np.float32)
    running_mean = np.array([2.5, 5.0], dtype=np.float32)
    running_var = np.array([1.25, 5.0], dtype=np.float32)
    params = {
        '_w_batchnorm2d_0': gamma,
        '_b_batchnorm2d_0': beta,
        '_running_mean_batchnorm2d_0': running_mean,
        '_running_var_batchnorm2d_0': running_var,
    }

    out = ForwardExecutor().run_inference(graph, x, params=params)
    expected = (
        (x - running_mean[None, :, None, None])
        / np.sqrt(running_var[None, :, None, None] + 1e-5)
    )
    expected = expected * gamma[None, :, None, None] + beta[None, :, None, None]

    assert out.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(out, expected.astype(np.float32), atol=1e-5)


def test_batchnorm2d_eval_defaults_to_identity_without_params():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    graph = build_graph([{'type': 'BatchNorm2d'}], (1, 1, 2, 2))
    x = np.array([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=np.float32)
    out = ForwardExecutor().run_inference(graph, x)
    np.testing.assert_allclose(out, x, atol=3e-5)


def test_batchnorm2d_train_mode_uses_batch_stats_and_updates_running_stats():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    graph = build_graph([{'type': 'BatchNorm2d', 'momentum': 0.25}], (1, 1, 2, 2))
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    params = {
        '_w_batchnorm2d_0': np.array([1.0], dtype=np.float32),
        '_b_batchnorm2d_0': np.array([0.0], dtype=np.float32),
        '_running_mean_batchnorm2d_0': np.array([0.0], dtype=np.float32),
        '_running_var_batchnorm2d_0': np.array([1.0], dtype=np.float32),
    }

    out = ForwardExecutor().run_inference(graph, x, params=params, mode='train')
    batch_mean = np.array([x.mean()], dtype=np.float32)
    batch_var = np.array([x.var()], dtype=np.float32)
    expected = (x - batch_mean.reshape(1, 1, 1, 1)) / np.sqrt(batch_var.reshape(1, 1, 1, 1) + 1e-5)

    np.testing.assert_allclose(out, expected.astype(np.float32), atol=1e-5)
    np.testing.assert_allclose(params['_running_mean_batchnorm2d_0'], 0.25 * batch_mean, atol=1e-6)
    np.testing.assert_allclose(
        params['_running_var_batchnorm2d_0'],
        (1.0 - 0.25) * np.array([1.0], dtype=np.float32) + 0.25 * batch_var,
        atol=1e-6,
    )


def test_batchnorm2d_invalid_mode_raises():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    graph = build_graph([{'type': 'BatchNorm2d'}], (1, 1, 2, 2))
    x = np.ones((1, 1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match='Unsupported cuda_native execution mode'):
        ForwardExecutor().run_inference(graph, x, mode='weird')
