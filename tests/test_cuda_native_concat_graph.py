from __future__ import annotations

import numpy as np


def test_build_graph_accepts_named_concat_branch():
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Concat', 'inputs': ['left', 'right'], 'axis': 1, 'output': 'cat'},
        ],
        (2, 3),
    )

    assert [node.op_type for node in graph.nodes] == ['Identity', 'Identity', 'Identity', 'Concat']
    assert graph.nodes[-1].inputs == ['left', 'right']
    assert graph.output_spec.name == 'cat'
    assert graph.output_spec.shape == (2, 6)


def test_forward_executor_runs_concat_branch_graph():
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1, 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Concat', 'inputs': ['left', 'right'], 'axis': 1, 'output': 'cat'},
        ],
        (2, 3, 5, 5),
    )

    params = {
        '_w_conv2d_0': np.ones((2, 3, 1, 1), dtype=np.float32),
        '_b_conv2d_0': np.zeros((2,), dtype=np.float32),
    }
    x = np.ones((2, 3, 5, 5), dtype=np.float32)

    out = ForwardExecutor().run_inference(graph, x, params=params)

    assert out.shape == (2, 4, 5, 5)
    assert np.allclose(out[:, :2], out[:, 2:])


def test_backward_executor_splits_gradients_across_concat_inputs():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Concat', 'inputs': ['left', 'right'], 'axis': 1, 'output': 'cat'},
        ],
        (2, 3),
    )
    x = np.ones((2, 3), dtype=np.float32)
    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x})

    grad_input, param_grads = BackwardExecutor().run(
        graph,
        np.ones((2, 6), dtype=np.float32),
        cache,
    )

    assert param_grads == {}
    assert grad_input.shape == (2, 3)
    assert np.allclose(grad_input, 2.0)


def test_full_config_validation_rejects_concat_shape_mismatch():
    from minicnn.cuda_native.api import validate_cuda_native_config

    errors = validate_cuda_native_config(
        {
            'engine': {'backend': 'cuda_native'},
            'dataset': {
                'type': 'random',
                'input_shape': [3, 8, 8],
                'num_classes': 2,
                'num_samples': 8,
                'val_samples': 4,
            },
            'model': {
                'layers': [
                    {'type': 'Identity', 'output': 'image'},
                    {'type': 'Flatten', 'inputs': ['image'], 'output': 'flat'},
                    {'type': 'Concat', 'inputs': ['image', 'flat'], 'axis': 1, 'output': 'cat'},
                ],
            },
            'train': {'batch_size': 2, 'epochs': 1},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
            'scheduler': {'enabled': False},
        }
    )

    assert errors
    assert any('same rank' in error or 'non-concat dimensions to match' in error for error in errors)
