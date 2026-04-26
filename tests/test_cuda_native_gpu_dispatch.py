from __future__ import annotations

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.gpu_bridge import (
    build_c_abi_kernel_trace,
    build_fixed_kernel_trace,
    build_flat_gpu_bridge_trace,
    build_gpu_bridge_trace,
)
from minicnn.cuda_native.gpu_dispatch import build_gpu_dispatch_plan, build_gpu_launch_trace
from minicnn.cuda_native.gpu_training_lowering import build_gpu_training_lowering_plan


def test_gpu_kernel_registry_marks_helper_backed_backward_ops_partial_native():
    from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs

    statuses = {spec.op_name: spec.backward_status for spec in list_gpu_kernel_specs()}
    helper_backed_backward_ops = {
        'AdaptiveAvgPool2d',
        'AvgPool2d',
        'BatchNorm2d',
        'Conv2d',
        'DepthwiseConv2d',
        'GELU',
        'GlobalAvgPool2d',
        'GroupNorm',
        'LayerNorm',
        'LayerNorm2d',
        'Linear',
        'MaxPool2d',
        'PointwiseConv2d',
        'ReLU',
        'Sigmoid',
        'SiLU',
        'Tanh',
    }

    assert {op: statuses[op] for op in helper_backed_backward_ops} == {
        op: 'partial_native' for op in helper_backed_backward_ops
    }
    assert statuses['Flatten'] == 'not_needed'
    assert statuses['Dropout'] == 'planned'
    assert statuses['DropPath'] == 'planned'


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
    assert summary['steps'][0]['forward_status'] == 'native_alias'
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
                {'type': 'Dropout', 'p': 0.1},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 1, 8, 8),
    )

    plan = build_gpu_dispatch_plan(graph)
    summary = plan.summary()

    assert summary['ready'] is False
    assert summary['unsupported_ops'] == ['Dropout']
    assert [step['op_name'] for step in summary['steps']] == ['Dropout', 'Flatten', 'Linear']
    assert summary['steps'][0]['supported'] is False
    assert summary['steps'][0]['launch_family'] == 'unsupported'
    assert summary['steps'][0]['lowering_kind'] == 'unsupported'
    assert summary['steps'][0]['launch_descriptor']['launch_family'] == 'unsupported'
    assert summary['steps'][0]['launch_descriptor']['input_shapes'] == [[1, 1, 8, 8]]
    assert summary['steps'][0]['launch_descriptor']['normalized_tensor_args'][0]['binding'] == 'input'
    assert summary['steps'][0]['forward_status'] == 'unsupported'
    assert summary['steps'][2]['param_keys'] == ['_w_linear_2', '_b_linear_2']


def test_gpu_dispatch_plan_supports_identity_and_noop_regularization_aliases():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Identity'},
                {'type': 'Dropout', 'p': 0.0},
                {'type': 'DropPath', 'p': 0.0},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 1, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert [step['op_name'] for step in summary['steps'][:3]] == ['Identity', 'Dropout', 'DropPath']
    assert [step['launch_family'] for step in summary['steps'][:3]] == ['identity_alias', 'identity_alias', 'identity_alias']
    assert [step['lowering_kind'] for step in summary['steps'][:3]] == [
        'shape_identity_alias_shim',
        'regularization_dropout_p0_alias_shim',
        'regularization_droppath_p0_alias_shim',
    ]
    assert summary['steps'][1]['launch_descriptor']['attr_bindings'] == {'p': 0.0}
    assert summary['steps'][2]['launch_descriptor']['attr_bindings'] == {'p': 0.0}


def test_gpu_dispatch_plan_supports_batchnorm2d_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'BatchNorm2d', 'num_features': 1, 'eps': 1e-4, 'momentum': 0.2},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 1, 8, 8),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'BatchNorm2d'
    assert summary['steps'][0]['launch_family'] == 'batchnorm2d_nchw'
    assert summary['steps'][0]['lowering_kind'] == 'normalization_batchnorm2d_shim'
    assert summary['steps'][0]['param_keys'] == [
        '_w_batchnorm2d_0',
        '_b_batchnorm2d_0',
        '_running_mean_batchnorm2d_0',
        '_running_var_batchnorm2d_0',
    ]
    assert summary['steps'][0]['launch_descriptor']['attr_bindings'] == {'eps': 0.0001, 'momentum': 0.2}


def test_gpu_bridge_trace_includes_batchnorm2d_attrs_in_bridge_payload():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'BatchNorm2d', 'num_features': 1, 'eps': 1e-4, 'momentum': 0.2},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 1, 8, 8),
    )

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)

    assert requests[0].op_name == 'BatchNorm2d'
    assert requests[0].bridge_payload['eps'] == 0.0001
    assert requests[0].bridge_payload['momentum'] == 0.2


def test_gpu_dispatch_plan_supports_global_avgpool_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'GlobalAvgPool2d'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 4, 8, 8),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'GlobalAvgPool2d'
    assert summary['steps'][0]['launch_family'] == 'global_avgpool2d_nchw'
    assert summary['steps'][0]['lowering_kind'] == 'pool_global_avgpool2d_shim'


def test_gpu_dispatch_plan_supports_avgpool_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 0},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 4, 8, 8),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'AvgPool2d'
    assert summary['steps'][0]['launch_family'] == 'avgpool2d_nchw'
    assert summary['steps'][0]['lowering_kind'] == 'pool_avgpool2d_shim'
    assert summary['steps'][0]['launch_descriptor']['attr_bindings'] == {
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
    }


def test_gpu_dispatch_plan_supports_adaptive_avgpool_output_size_one_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'AdaptiveAvgPool2d', 'output_size': 1},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 4, 8, 8),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'AdaptiveAvgPool2d'
    assert summary['steps'][0]['launch_family'] == 'global_avgpool2d_nchw'
    assert summary['steps'][0]['launch_descriptor']['attr_bindings'] == {'output_size': 1}


def test_gpu_dispatch_plan_supports_modern_elementwise_activation_shims():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'GELU'},
                {'type': 'SiLU'},
                {'type': 'Sigmoid'},
                {'type': 'Tanh'},
            ],
        },
        (1, 3, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert [step['op_name'] for step in summary['steps']] == ['GELU', 'SiLU', 'Sigmoid', 'Tanh']
    assert [step['lowering_kind'] for step in summary['steps']] == [
        'activation_gelu_shim',
        'activation_silu_shim',
        'activation_sigmoid_shim',
        'activation_tanh_shim',
    ]


def test_gpu_dispatch_plan_supports_pointwise_conv_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'PointwiseConv2d', 'out_channels': 2, 'bias': False},
                {'type': 'GELU'},
                {'type': 'GlobalAvgPool2d'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 3, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'PointwiseConv2d'
    assert summary['steps'][0]['launch_family'] == 'conv2d_nchw'
    assert summary['steps'][0]['param_keys'] == ['_w_pointwiseconv2d_0']
    assert summary['steps'][0]['launch_descriptor']['attr_bindings']['dilation'] == 1


def test_gpu_dispatch_plan_supports_depthwise_conv_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'DepthwiseConv2d', 'out_channels': 3, 'kernel_size': 3, 'padding': 1, 'bias': False},
                {'type': 'GELU'},
                {'type': 'PointwiseConv2d', 'out_channels': 2, 'bias': False},
            ],
        },
        (1, 3, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'DepthwiseConv2d'
    assert summary['steps'][0]['launch_family'] == 'depthwise_conv2d_nchw'
    assert summary['steps'][0]['lowering_kind'] == 'depthwise_conv2d_shim'
    assert summary['steps'][0]['param_keys'] == ['_w_depthwiseconv2d_0']
    assert summary['steps'][0]['launch_descriptor']['attr_bindings']['groups'] == 3


def test_gpu_bridge_trace_includes_depthwise_conv_payload_geometry():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'DepthwiseConv2d', 'out_channels': 3, 'kernel_size': 3, 'padding': 1, 'bias': False},
            ],
        },
        (1, 3, 4, 4),
    )

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)

    assert requests[0].op_name == 'DepthwiseConv2d'
    assert requests[0].bridge_payload['in_channels'] == 3
    assert requests[0].bridge_payload['out_channels'] == 3
    assert requests[0].bridge_payload['input_hw'] == [4, 4]
    assert requests[0].bridge_payload['output_hw'] == [4, 4]
    assert requests[0].bridge_payload['dilation'] == 1
    assert requests[0].bridge_payload['groups'] == 3


def test_gpu_bridge_trace_includes_conv_dilation_in_payload():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'dilation': 2, 'bias': False},
            ],
        },
        (1, 3, 8, 8),
    )

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)

    assert requests[0].op_name == 'Conv2d'
    assert requests[0].bridge_payload['dilation'] == 2


def test_gpu_dispatch_plan_supports_layernorm2d_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'LayerNorm2d', 'num_channels': 3, 'eps': 1e-5},
                {'type': 'GELU'},
                {'type': 'GlobalAvgPool2d'},
            ],
        },
        (1, 3, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'LayerNorm2d'
    assert summary['steps'][0]['launch_family'] == 'layernorm2d_nchw'
    assert summary['steps'][0]['lowering_kind'] == 'normalization_layernorm2d_shim'
    assert summary['steps'][0]['param_keys'] == ['_w_layernorm2d_0', '_b_layernorm2d_0']


def test_gpu_dispatch_plan_supports_layernorm_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'LayerNorm', 'normalized_shape': 16, 'eps': 1e-5},
                {'type': 'GELU'},
            ],
        },
        (3, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][1]['op_name'] == 'LayerNorm'
    assert summary['steps'][1]['launch_family'] == 'layernorm_nd'
    assert summary['steps'][1]['lowering_kind'] == 'normalization_layernorm_shim'
    assert summary['steps'][1]['param_keys'] == ['_w_layernorm_1', '_b_layernorm_1']
    assert summary['steps'][1]['launch_descriptor']['attr_bindings'] == {'eps': 1e-05, 'normalized_shape': 16}
    assert summary['steps'][1]['launch_descriptor']['param_layouts'] == {
        '_w_layernorm_1': 'normalized_shape',
        '_b_layernorm_1': 'normalized_shape',
    }


def test_gpu_dispatch_plan_supports_groupnorm_forward_shim():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'GroupNorm', 'num_groups': 2, 'eps': 1e-5},
                {'type': 'GELU'},
                {'type': 'GlobalAvgPool2d'},
            ],
        },
        (1, 4, 4, 4),
    )

    summary = build_gpu_dispatch_plan(graph).summary()

    assert summary['ready'] is True
    assert summary['unsupported_ops'] == []
    assert summary['steps'][0]['op_name'] == 'GroupNorm'
    assert summary['steps'][0]['launch_family'] == 'groupnorm_nchw'
    assert summary['steps'][0]['lowering_kind'] == 'normalization_groupnorm_shim'
    assert summary['steps'][0]['param_keys'] == ['_w_groupnorm_0', '_b_groupnorm_0']
    assert summary['steps'][0]['launch_descriptor']['attr_bindings'] == {'num_groups': 2, 'eps': 1e-05}


def test_gpu_training_lowering_plan_records_linear_rmsprop_manifest():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ],
        },
        (1, 8, 8),
    )

    plan = build_gpu_training_lowering_plan(
        graph,
        loss_cfg={'type': 'MSELoss'},
        optim_cfg={'type': 'RMSprop', 'alpha': 0.9, 'momentum': 0.1},
        train_cfg={'grad_accum_steps': 1, 'amp': False},
    )
    summary = plan.summary()

    assert summary['ready'] is True
    assert summary['subset_name'] == 'flatten_linear'
    assert summary['helper'] == 'native_gpu_linear_training_step'
    assert [step['op_name'] for step in summary['forward_steps']] == ['Flatten', 'Linear']
    assert summary['loss_step']['lowering_kind'] == 'mse_fwd_grad_loss_acc'
    assert summary['backward_steps'][0]['lowering_kind'] == 'dense_backward_full'
    assert summary['optimizer_steps'][0]['lowering_kind'] == 'rmsprop_update_fused'
    assert summary['optimizer_steps'][0]['param_keys'] == ['_w_linear_1', '_b_linear_1']
    assert summary['unsupported_reasons'] == []


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


def test_conv_fixed_and_c_abi_bridge_trace_carry_dilation():
    graph = build_cuda_native_graph(
        {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'dilation': 2, 'bias': False},
            ],
        },
        (1, 3, 8, 8),
    )

    packets = build_gpu_launch_trace(build_gpu_dispatch_plan(graph))
    requests = build_gpu_bridge_trace(packets)
    flat_requests = build_flat_gpu_bridge_trace(requests)
    fixed_calls = build_fixed_kernel_trace(flat_requests)
    c_abi_calls = build_c_abi_kernel_trace(fixed_calls)

    assert fixed_calls[0].dilation_h == 2
    assert fixed_calls[0].dilation_w == 2
    assert c_abi_calls[0].int_args8 == (1, 1, 0, 0, 1, 2, 2, 0)


def test_c_abi_gpu_bridge_trace_builds_stable_records():
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
    c_abi_calls = build_c_abi_kernel_trace(fixed_calls)

    assert len(c_abi_calls) == 4
    assert c_abi_calls[1].op_name == 'Linear'
    assert c_abi_calls[1].op_code == 2
    assert c_abi_calls[1].launch_family_code == 2
    assert c_abi_calls[1].dtype_code == 1
    assert c_abi_calls[1].preferred_layout_code == 1
    assert c_abi_calls[1].input_shape4 == (1, 64, 1, 1)
    assert c_abi_calls[1].output_shape4 == (1, 8, 1, 1)
    assert c_abi_calls[1].int_args8 == (0, 0, 0, 0, 1, 1, 64, 8)
    assert c_abi_calls[1].flags[:2] == (1, 1)
