from __future__ import annotations

import numpy as np


def test_layernorm_kernel_preserves_shape():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.unified._cuda_native_bridge import _init_params

    graph = build_graph(
        [{'type': 'LayerNorm', 'normalized_shape': [4, 5]}],
        (2, 3, 4, 5),
    )
    params = _init_params(graph, seed=0)
    x = np.random.default_rng(0).standard_normal((2, 3, 4, 5)).astype(np.float32)
    ctx = ForwardExecutor().run(graph, {'input': x}, params=params)
    assert ctx[graph.output_spec.name].shape == x.shape


def test_layernorm_backward_returns_input_and_param_grads():
    from minicnn.cuda_native.backward import _bwd_layernorm
    from minicnn.cuda_native.nodes import Node

    node = Node(
        name='ln0',
        op_type='LayerNorm',
        attrs={'normalized_shape': [4, 5], 'eps': 1e-5},
        inputs=['x'],
        outputs=['y'],
    )
    x = np.random.default_rng(1).standard_normal((2, 3, 4, 5)).astype(np.float32)
    grad_out = np.random.default_rng(2).standard_normal((2, 3, 4, 5)).astype(np.float32)
    cache = {
        'fwd_ln0_in': x,
        '_w_ln0': np.ones((4, 5), dtype=np.float32),
        '_b_ln0': np.zeros((4, 5), dtype=np.float32),
    }
    param_grads: dict[str, np.ndarray] = {}

    grad_in = _bwd_layernorm(node, grad_out, cache, param_grads)

    assert grad_in.shape == x.shape
    assert param_grads['_w_ln0'].shape == (4, 5)
    assert param_grads['_b_ln0'].shape == (4, 5)
