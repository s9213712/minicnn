from __future__ import annotations

import numpy as np

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_executor import GpuStubExecutor
from minicnn.unified._cuda_native_bridge import _init_params


def test_gpu_stub_executor_runs_bootstrap_subset_graph():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 8},
                {'type': 'ReLU'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 8, 8),
    )
    params = _init_params(graph, seed=7)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu')
    runtime.reserve_from_planner(total_bytes=4096, num_buffers=4)
    executor = GpuStubExecutor(device_runtime=runtime)

    result = executor.run(graph, np.ones((1, 8, 8), dtype=np.float32), params=params)
    summary = result.summary()
    runtime_summary = runtime.summary()

    assert summary['output_name'] == graph.output_spec.name
    assert summary['dispatch_plan']['ready'] is True
    assert summary['dispatch_plan']['num_steps'] == 4
    assert tuple(summary['output_shape']) == (1, 2)
    assert runtime_summary['tensor_execution_device'] == 'gpu'
    assert runtime_summary['execution_kinds']['gpu_stub_forward'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_kernel:Flatten'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_kernel:Linear'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_kernel:ReLU'] == 1
    assert runtime_summary['reserved_buffer_reuse_events'] >= 2
    assert runtime_summary['reserved_buffer_release_events'] >= 2


def test_gpu_stub_executor_rejects_graph_outside_bootstrap_subset():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'BatchNorm2d', 'num_features': 1},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 1, 8, 8),
    )
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu')
    runtime.reserve_from_planner(total_bytes=4096, num_buffers=4)
    executor = GpuStubExecutor(device_runtime=runtime)

    try:
        executor.run(graph, np.ones((1, 1, 8, 8), dtype=np.float32))
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError('expected bootstrap-subset rejection')

    assert 'unsupported_ops' in message
    assert 'BatchNorm2d' in message
