from __future__ import annotations

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.gpu_bridge import build_fixed_kernel_trace, build_flat_gpu_bridge_trace, build_gpu_bridge_trace
from minicnn.cuda_native.gpu_dispatch import build_gpu_dispatch_plan, build_gpu_launch_trace


def test_gpu_dispatch_plan_supports_bootstrap_subset_graph():
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

    plan = build_gpu_dispatch_plan(graph)
    summary = plan.summary()

    assert summary['execution_mode'] == 'gpu_native'
    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert [step['op_name'] for step in summary['steps']] == ['Flatten', 'Linear', 'ReLU', 'Linear']
    assert summary['steps'][0]['launch_family'] == 'reshape_view'
    assert summary['steps'][0]['forward_status'] == 'planned'
    assert summary['steps'][0]['backward_status'] == 'not_needed'
    assert summary['steps'][0]['input_arity'] == 1
    assert summary['steps'][0]['output_arity'] == 1
    assert summary['steps'][0]['preferred_layout'] == 'row_major'
    assert summary['steps'][0]['param_keys'] == []
    assert summary['steps'][0]['lowering_kind'] == 'shape_flatten_shim'
    assert summary['steps'][0]['launch_descriptor']['input_bindings'] == ['input']
    assert summary['steps'][0]['launch_descriptor']['output_bindings'] == ['t_1']
    assert summary['steps'][0]['launch_descriptor']['param_bindings'] == []
    assert summary['steps'][0]['launch_descriptor']['attr_bindings'] == {}
    assert summary['steps'][0]['launch_descriptor']['input_shapes'] == [[1, 8, 8]]
    assert summary['steps'][0]['launch_descriptor']['output_shapes'] == [[1, 64]]
    assert summary['steps'][0]['launch_descriptor']['tensor_dtype'] == 'float32'
    assert summary['steps'][0]['launch_descriptor']['param_layouts'] == {}
    assert summary['steps'][0]['launch_descriptor']['normalized_tensor_args'] == [
        {
            'kind': 'input',
            'index': 0,
            'binding': 'input',
            'shape': [1, 8, 8],
            'dtype': 'float32',
            'layout': 'NCHW',
        },
        {
            'kind': 'output',
            'index': 0,
            'binding': 't_1',
            'shape': [1, 64],
            'dtype': 'float32',
            'layout': 'NCHW',
        },
    ]
    assert summary['steps'][0]['launch_descriptor']['normalized_scalar_args'] == []
    assert summary['steps'][0]['supported'] is True
    assert summary['steps'][1]['category'] == 'linear'
    assert summary['steps'][1]['launch_family'] == 'gemm_affine'
    assert summary['steps'][1]['param_keys'] == ['_w_linear_1', '_b_linear_1']
    assert summary['steps'][1]['lowering_kind'] == 'linear_affine_shim'
    assert summary['steps'][1]['launch_descriptor']['param_bindings'] == ['_w_linear_1', '_b_linear_1']
    assert summary['steps'][1]['launch_descriptor']['input_shapes'] == [[1, 64]]
    assert summary['steps'][1]['launch_descriptor']['output_shapes'] == [[1, 8]]
    assert summary['steps'][1]['launch_descriptor']['param_layouts'] == {
        '_w_linear_1': 'OI',
        '_b_linear_1': 'O',
    }
    assert summary['steps'][1]['launch_descriptor']['normalized_tensor_args'] == [
        {
            'kind': 'input',
            'index': 0,
            'binding': 't_1',
            'shape': [1, 64],
            'dtype': 'float32',
            'layout': 'NCHW',
        },
        {
            'kind': 'output',
            'index': 0,
            'binding': 't_2',
            'shape': [1, 8],
            'dtype': 'float32',
            'layout': 'NCHW',
        },
        {
            'kind': 'param',
            'index': 0,
            'binding': '_w_linear_1',
            'layout': 'OI',
        },
        {
            'kind': 'param',
            'index': 1,
            'binding': '_b_linear_1',
            'layout': 'O',
        },
    ]
    assert summary['steps'][1]['launch_descriptor']['normalized_scalar_args'] == []


def test_gpu_dispatch_plan_marks_ops_outside_bootstrap_subset():
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

    plan = build_gpu_dispatch_plan(graph)
    summary = plan.summary()

    assert summary['ready'] is False
    assert summary['unsupported_ops'] == ['BatchNorm2d']
    assert [step['op_name'] for step in summary['steps']] == ['BatchNorm2d', 'Flatten', 'Linear']
    assert summary['steps'][0]['supported'] is False
    assert summary['steps'][0]['launch_family'] == 'unsupported'
    assert summary['steps'][0]['lowering_kind'] == 'unsupported'
    assert summary['steps'][0]['launch_descriptor']['launch_family'] == 'unsupported'
    assert summary['steps'][0]['launch_descriptor']['input_shapes'] == [[1, 1, 8, 8]]
    assert summary['steps'][0]['launch_descriptor']['normalized_tensor_args'][0]['binding'] == 'input'
    assert summary['steps'][0]['forward_status'] == 'unsupported'
    assert summary['steps'][2]['param_keys'] == ['_w_linear_2', '_b_linear_2']


def test_gpu_launch_trace_builds_normalized_packets():
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

    plan = build_gpu_dispatch_plan(graph)
    packets = build_gpu_launch_trace(plan)

    assert len(packets) == 4
    assert packets[0].launch_family == 'reshape_view'
    assert packets[1].launch_family == 'gemm_affine'
    assert packets[1].tensor_args[0]['binding'] == 't_1'
    assert packets[1].tensor_args[2]['binding'] == '_w_linear_1'
    assert packets[1].tensor_args[2]['layout'] == 'OI'
    assert packets[1].scalar_args == ()


def test_gpu_bridge_trace_builds_stub_requests():
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

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)

    assert len(requests) == 4
    assert requests[0].request_id == 'flatten_0:0'
    assert requests[1].dispatch_mode == 'gpu_bridge_stub'
    assert requests[1].launch_family == 'gemm_affine'
    assert requests[1].tensor_args[2]['binding'] == '_w_linear_1'
    assert requests[1].bridge_payload == {
        'op_name': 'Linear',
        'launch_family': 'gemm_affine',
        'preferred_layout': 'row_major',
        'input_shape': [1, 64],
        'output_shape': [1, 8],
        'tensor_dtype': 'float32',
        'matmul_m': 1,
        'matmul_k': 64,
        'matmul_n': 8,
        'weight_binding': '_w_linear_1',
        'weight_layout': 'OI',
        'has_bias': True,
    }


def test_flat_gpu_bridge_trace_builds_flat_requests():
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

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)
    flat_requests = build_flat_gpu_bridge_trace(requests)

    assert len(flat_requests) == 4
    assert flat_requests[1].launch_family == 'gemm_affine'
    assert flat_requests[1].tensor_bindings == ('t_1', 't_2', '_w_linear_1', '_b_linear_1')
    assert flat_requests[1].tensor_roles == ('input', 'output', 'param', 'param')
    assert flat_requests[1].param_bindings == ('_w_linear_1', '_b_linear_1')
    assert flat_requests[1].bridge_payload['matmul_k'] == 64


def test_fixed_gpu_bridge_trace_builds_fixed_calls():
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

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)
    flat_requests = build_flat_gpu_bridge_trace(requests)
    fixed_calls = build_fixed_kernel_trace(flat_requests)

    assert len(fixed_calls) == 4
    assert fixed_calls[1].launch_family == 'gemm_affine'
    assert fixed_calls[1].dispatch_mode == 'gpu_fixed_bridge_stub'
    assert fixed_calls[1].input_binding == 't_1'
    assert fixed_calls[1].output_binding == 't_2'
    assert fixed_calls[1].weight_binding == '_w_linear_1'
    assert fixed_calls[1].bias_binding == '_b_linear_1'
    assert fixed_calls[1].matmul_m == 1
    assert fixed_calls[1].matmul_k == 64
    assert fixed_calls[1].matmul_n == 8
