from __future__ import annotations

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.gpu_dispatch import build_gpu_dispatch_plan


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
    assert summary['steps'][0]['supported'] is True
    assert summary['steps'][1]['category'] == 'linear'
    assert summary['steps'][1]['launch_family'] == 'gemm_affine'
    assert summary['steps'][1]['param_keys'] == ['_w_linear_1', '_b_linear_1']
    assert summary['steps'][1]['lowering_kind'] == 'linear_affine_shim'
    assert summary['steps'][1]['launch_descriptor']['param_bindings'] == ['_w_linear_1', '_b_linear_1']


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
    assert summary['steps'][0]['forward_status'] == 'unsupported'
    assert summary['steps'][2]['param_keys'] == ['_w_linear_2', '_b_linear_2']
