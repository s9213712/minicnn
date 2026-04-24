from __future__ import annotations

import ctypes

import numpy as np

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_bridge_adapter import GpuNativeLibraryBridgeAdapter
from minicnn.cuda_native.gpu_executor import GpuStubExecutor
from minicnn.unified._cuda_native_bridge import _init_params


class _FakeCudaLib:
    def __init__(self):
        self.next_ptr = 1000
        self.memory = {}
        self.dense_forward_calls = []

    def gpu_malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 8
        self.memory[ptr] = np.zeros(int(nbytes) // 4, dtype=np.float32)
        return ptr

    def gpu_free(self, ptr):
        self.memory.pop(ptr, None)

    def gpu_memcpy_h2d(self, ptr, host_ptr, nbytes):
        count = int(nbytes) // 4
        src = np.ctypeslib.as_array((ctypes.c_float * count).from_address(int(host_ptr)))
        self.memory[ptr][...] = src

    def gpu_memcpy_d2h(self, host_ptr, ptr, nbytes):
        count = int(nbytes) // 4
        dst = np.ctypeslib.as_array((ctypes.c_float * count).from_address(int(host_ptr)))
        dst[...] = self.memory[ptr]

    def dense_forward(self, d_input, d_weight, d_bias, d_output, n, in_f, out_f):
        self.dense_forward_calls.append((d_input, d_weight, d_bias, d_output, n, in_f, out_f))
        x = self.memory[d_input].reshape(int(n), int(in_f))
        w = self.memory[d_weight].reshape(int(out_f), int(in_f))
        b = self.memory[d_bias].reshape(int(out_f))
        self.memory[d_output][...] = (x @ w.T + b).reshape(-1)


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
    assert len(summary['launch_trace']) == 4
    assert len(summary['bridge_trace']) == 4
    assert len(summary['flat_bridge_trace']) == 4
    assert len(summary['fixed_bridge_trace']) == 4
    assert len(summary['c_abi_bridge_trace']) == 4
    assert len(summary['bridge_results']) == 4
    assert len(summary['flat_bridge_results']) == 4
    assert len(summary['fixed_bridge_results']) == 4
    assert len(summary['c_abi_bridge_results']) == 4
    assert summary['launch_trace'][0]['launch_family'] == 'reshape_view'
    assert summary['launch_trace'][1]['launch_family'] == 'gemm_affine'
    assert summary['launch_trace'][1]['tensor_args'][2]['binding'] == '_w_linear_1'
    assert summary['bridge_trace'][1]['dispatch_mode'] == 'gpu_bridge_stub'
    assert summary['bridge_trace'][1]['launch_family'] == 'gemm_affine'
    assert summary['bridge_trace'][1]['tensor_args'][2]['binding'] == '_w_linear_1'
    assert summary['bridge_trace'][1]['bridge_payload']['matmul_n'] == 8
    assert summary['bridge_results'][1]['accepted'] is True
    assert summary['bridge_results'][1]['dispatch_mode'] == 'gpu_bridge_stub'
    assert summary['flat_bridge_trace'][1]['launch_family'] == 'gemm_affine'
    assert summary['flat_bridge_trace'][1]['tensor_bindings'] == ['t_1', 't_2', '_w_linear_1', '_b_linear_1']
    assert summary['flat_bridge_results'][1]['accepted'] is True
    assert summary['flat_bridge_results'][1]['flat_tensor_arg_count'] == 4
    assert summary['fixed_bridge_trace'][1]['launch_family'] == 'gemm_affine'
    assert summary['fixed_bridge_trace'][1]['weight_binding'] == '_w_linear_1'
    assert summary['fixed_bridge_trace'][1]['matmul_k'] == 64
    assert summary['fixed_bridge_results'][1]['accepted'] is True
    assert summary['fixed_bridge_results'][1]['dispatch_mode'] == 'gpu_backend_stub'
    assert summary['fixed_bridge_results'][1]['abi_version'] == 1
    assert summary['fixed_bridge_results'][1]['kernel_symbol'] == 'minicnn_gpu_linear_f32'
    assert summary['fixed_bridge_results'][1]['matmul_signature'] == [1, 64, 8]
    assert summary['c_abi_bridge_trace'][1]['op_code'] == 2
    assert summary['c_abi_bridge_trace'][1]['launch_family_code'] == 2
    assert summary['c_abi_bridge_trace'][1]['int_args8'] == [0, 0, 0, 0, 1, 1, 64, 8]
    assert summary['c_abi_bridge_results'][1]['dispatch_mode'] == 'gpu_c_abi_stub'
    assert summary['c_abi_bridge_results'][1]['abi_version'] == 1
    assert tuple(summary['output_shape']) == (1, 2)
    assert runtime_summary['tensor_execution_device'] == 'gpu'
    assert runtime_summary['execution_kinds']['gpu_stub_forward'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_kernel:Flatten'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_kernel:Linear'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_kernel:ReLU'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_launch:reshape_view'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_launch:gemm_affine'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_launch:elementwise_unary'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_packet:reshape_view'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_packet:gemm_affine'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_packet:elementwise_unary'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_bridge:reshape_view'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_bridge:gemm_affine'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_bridge:elementwise_unary'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_flat_bridge:reshape_view'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_flat_bridge:gemm_affine'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_flat_bridge:elementwise_unary'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_fixed_bridge:reshape_view'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_fixed_bridge:gemm_affine'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_fixed_bridge:elementwise_unary'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_c_abi_bridge:reshape_view'] == 1
    assert runtime_summary['execution_kinds']['gpu_stub_c_abi_bridge:gemm_affine'] == 2
    assert runtime_summary['execution_kinds']['gpu_stub_c_abi_bridge:elementwise_unary'] == 1
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


def test_gpu_stub_executor_backend_stub_routes_conv2d():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'padding': 1},
                {'type': 'ReLU'},
                {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 3, 8, 8),
    )
    params = _init_params(graph, seed=11)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu')
    runtime.reserve_from_planner(total_bytes=8192, num_buffers=8)
    executor = GpuStubExecutor(device_runtime=runtime)

    result = executor.run(graph, np.ones((1, 3, 8, 8), dtype=np.float32), params=params)
    summary = result.summary()

    conv_result = summary['fixed_bridge_results'][0]
    assert conv_result['dispatch_mode'] == 'gpu_backend_stub'
    assert conv_result['kernel_symbol'] == 'minicnn_gpu_conv2d_nchw_f32'
    assert conv_result['input_shape'] == [1, 3, 8, 8]
    assert conv_result['output_shape'] == [1, 4, 8, 8]
    assert conv_result['stride'] == [1, 1]
    assert conv_result['padding'] == [1, 1]


def test_gpu_stub_executor_can_use_native_library_bridge_adapter():
    class _FakeLib:
        def dense_forward(self):
            raise AssertionError('symbol availability check must not execute kernel')

    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 4),
    )
    params = _init_params(graph, seed=13)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu')
    runtime.reserve_from_planner(total_bytes=4096, num_buffers=4)
    adapter = GpuNativeLibraryBridgeAdapter(bound_lib=_FakeLib())
    executor = GpuStubExecutor(device_runtime=runtime, c_abi_bridge_adapter=adapter)

    result = executor.run(graph, np.ones((1, 4), dtype=np.float32), params=params)
    summary = result.summary()

    linear_result = summary['c_abi_bridge_results'][1]
    assert linear_result['dispatch_mode'] == 'gpu_native_library_bridge'
    assert linear_result['kernel_symbol'] == 'dense_forward'
    assert linear_result['native_library_loaded'] is True
    assert linear_result['symbol_available'] is True
    assert linear_result['requires_device_pointers'] is True
    assert linear_result['executed'] is False


def test_gpu_stub_executor_uses_native_dense_forward_when_device_pointers_available():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 4),
    )
    params = {
        '_w_linear_1': np.asarray([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 2.0]], dtype=np.float32),
        '_b_linear_1': np.asarray([0.25, -0.5], dtype=np.float32),
    }
    fake_lib = _FakeCudaLib()
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=fake_lib,
    )
    runtime.reserve_from_planner(total_bytes=4096, num_buffers=4)
    executor = GpuStubExecutor(device_runtime=runtime)

    x = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result = executor.run(graph, x, params=params)

    expected = x @ params['_w_linear_1'].T + params['_b_linear_1']
    np.testing.assert_allclose(result.output, expected.astype(np.float32))
    assert len(fake_lib.dense_forward_calls) == 1

    summary = runtime.summary()
    assert summary['native_device_pointers_enabled'] is True
    assert summary['execution_kinds']['gpu_native_kernel:dense_forward'] == 1
    assert summary['device_pointer_allocation_events'] >= 4
    assert summary['device_sync_to_device_events'] >= 3
    assert summary['device_sync_to_host_events'] >= 1
