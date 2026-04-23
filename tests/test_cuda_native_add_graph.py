from __future__ import annotations

import numpy as np


def test_build_graph_accepts_named_add_branch():
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
        ],
        (2, 3),
    )

    assert [node.op_type for node in graph.nodes] == ['Identity', 'Identity', 'Identity', 'Add']
    assert graph.nodes[-1].inputs == ['left', 'right']
    assert graph.output_spec.name == 'sum'
    assert graph.output_spec.shape == (2, 3)


def test_forward_executor_runs_add_branch_graph():
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 1, 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'skip'},
            {'type': 'Conv2d', 'inputs': ['stem'], 'out_channels': 4, 'kernel_size': 1, 'output': 'main'},
            {'type': 'Add', 'inputs': ['skip', 'main'], 'output': 'sum'},
        ],
        (2, 3, 5, 5),
    )

    params = {
        '_w_conv2d_0': np.ones((4, 3, 1, 1), dtype=np.float32),
        '_b_conv2d_0': np.zeros((4,), dtype=np.float32),
        '_w_conv2d_2': np.ones((4, 4, 1, 1), dtype=np.float32),
        '_b_conv2d_2': np.zeros((4,), dtype=np.float32),
    }
    x = np.ones((2, 3, 5, 5), dtype=np.float32)

    out = ForwardExecutor().run_inference(graph, x, params=params)

    assert out.shape == (2, 4, 5, 5)
    assert np.allclose(out, 15.0)


def test_backward_executor_accumulates_gradients_across_add_inputs():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
        ],
        (2, 3),
    )
    x = np.ones((2, 3), dtype=np.float32)
    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x})

    grad_input, param_grads = BackwardExecutor().run(
        graph,
        np.ones((2, 3), dtype=np.float32),
        cache,
    )

    assert param_grads == {}
    assert grad_input.shape == (2, 3)
    assert np.allclose(grad_input, 2.0)


def test_validator_rejects_add_without_two_inputs():
    from minicnn.cuda_native.validators import validate_cuda_native_model_config

    errors = validate_cuda_native_model_config(
        {
            'layers': [
                {'type': 'Identity', 'output': 'stem'},
                {'type': 'Add', 'inputs': ['stem']},
            ]
        }
    )

    assert errors
    assert 'at least two tensor names' in errors[0]


def test_full_config_validation_rejects_unknown_add_input():
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
                    {'type': 'Identity', 'output': 'stem'},
                    {'type': 'Add', 'inputs': ['stem', 'missing'], 'output': 'sum'},
                ],
            },
            'train': {'batch_size': 2, 'epochs': 1},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
            'scheduler': {'enabled': False},
        }
    )

    assert errors
    assert any('unknown input tensor' in error for error in errors)


def test_full_config_validation_rejects_add_shape_mismatch():
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
                    {'type': 'Add', 'inputs': ['image', 'flat'], 'output': 'sum'},
                ],
            },
            'train': {'batch_size': 2, 'epochs': 1},
            'optimizer': {'type': 'SGD', 'lr': 0.01},
            'loss': {'type': 'CrossEntropyLoss'},
            'scheduler': {'enabled': False},
        }
    )

    assert errors
    assert any('expects all input shapes to match' in error for error in errors)
