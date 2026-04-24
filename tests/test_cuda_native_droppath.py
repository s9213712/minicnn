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
