from __future__ import annotations

import numpy as np


def test_droppath_kernel_preserves_shape_in_eval():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    graph = build_graph([{'type': 'DropPath', 'p': 0.2}], (4, 8, 8, 8))
    x = np.random.default_rng(0).standard_normal((4, 8, 8, 8)).astype(np.float32)
    ctx = ForwardExecutor().run(graph, {'input': x}, mode='eval')
    out = ctx[graph.output_spec.name]
    assert out.shape == x.shape
    assert np.allclose(out, x)


def test_droppath_backward_uses_cached_mask():
    from minicnn.cuda_native.backward import _bwd_droppath
    from minicnn.cuda_native.nodes import Node

    node = Node(name='dp0', op_type='DropPath', attrs={'p': 0.2}, inputs=['x'], outputs=['y'])
    grad_out = np.ones((2, 3, 4, 4), dtype=np.float32)
    mask = np.zeros_like(grad_out, dtype=np.float32)
    mask[0, ...] = 1.25
    cache = {'__cache_dp0': {'mask': mask}}

    grad_in = _bwd_droppath(node, grad_out, cache, {})

    assert grad_in.shape == grad_out.shape
    assert np.allclose(grad_in, mask)


def test_droppath_train_forward_matches_deterministic_reference_mask():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    graph = build_graph([{'type': 'DropPath', 'p': 0.2}], (4, 3, 2, 2))
    x = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)

    out = ForwardExecutor().run_inference(graph, x, mode='train')

    rng = np.random.default_rng(42)
    keep_prob = 0.8
    mask = (rng.random((4, 1, 1, 1), dtype=np.float32) < keep_prob).astype(np.float32) / keep_prob
    mask = np.broadcast_to(mask, x.shape).astype(np.float32)
    expected = (x * mask).astype(np.float32)

    assert np.allclose(out, expected)


def test_droppath_train_backward_matches_cached_forward_mask():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph([{'type': 'DropPath', 'p': 0.2}], (4, 3, 2, 2))
    x = np.random.default_rng(1).standard_normal((4, 3, 2, 2)).astype(np.float32)
    grad_out = np.random.default_rng(2).standard_normal((4, 3, 2, 2)).astype(np.float32)

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, mode='train')
    grad_in, param_grads = BackwardExecutor().run(graph, grad_out, cache)

    expected = grad_out * cache['__cache_droppath_0']['mask']

    assert param_grads == {}
    assert np.allclose(grad_in, expected.astype(np.float32))
