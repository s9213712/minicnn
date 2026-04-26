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


def test_gpu_stub_executor_dispatches_layernorm_via_gpu_lowering():
    from minicnn.cuda_native.api import build_cuda_native_graph
    from minicnn.cuda_native.device_runtime import DeviceRuntime
    from minicnn.cuda_native.gpu_executor import GpuStubExecutor
    from minicnn.unified._cuda_native_bridge import _init_params

    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'LayerNorm', 'normalized_shape': [4, 5], 'eps': 1e-5},
            ],
        },
        (3, 4, 5),
    )
    params = _init_params(graph, seed=0)
    x = np.random.default_rng(3).standard_normal((2, 3, 4, 5)).astype(np.float32)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu')
    runtime.reserve_from_planner(total_bytes=4096, num_buffers=4)

    result = GpuStubExecutor(device_runtime=runtime).run(graph, x, params=params)
    summary = runtime.summary()

    assert result.output.shape == x.shape
    assert summary['execution_kinds']['gpu_stub_kernel:LayerNorm'] == 1
    assert summary['execution_kinds']['gpu_stub_forward'] == 1


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
