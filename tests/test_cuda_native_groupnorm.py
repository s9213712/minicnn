from __future__ import annotations

import numpy as np


def test_groupnorm_forward_preserves_shape():
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'GroupNorm', 'num_groups': 2}],
        (2, 4, 3, 3),
    )
    params = {
        '_w_groupnorm_0': np.ones((4,), dtype=np.float32),
        '_b_groupnorm_0': np.zeros((4,), dtype=np.float32),
    }
    x = np.arange(72, dtype=np.float32).reshape(2, 4, 3, 3)

    out = ForwardExecutor().run_inference(graph, x, params=params)

    assert out.shape == x.shape


def test_groupnorm_backward_returns_input_grad_and_param_grads():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'GroupNorm', 'num_groups': 2}],
        (2, 4, 3, 3),
    )
    params = {
        '_w_groupnorm_0': np.ones((4,), dtype=np.float32),
        '_b_groupnorm_0': np.zeros((4,), dtype=np.float32),
    }
    x = np.arange(72, dtype=np.float32).reshape(2, 4, 3, 3)
    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params)

    grad_input, param_grads = BackwardExecutor().run(
        graph,
        np.ones_like(x, dtype=np.float32),
        cache,
    )

    assert grad_input.shape == x.shape
    assert param_grads['_w_groupnorm_0'].shape == (4,)
    assert param_grads['_b_groupnorm_0'].shape == (4,)
