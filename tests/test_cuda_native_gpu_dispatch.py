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
    assert summary['steps'][0]['forward_status'] == 'planned'
    assert summary['steps'][0]['backward_status'] == 'not_needed'
    assert summary['steps'][1]['category'] == 'linear'


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
    assert [step['op_name'] for step in summary['steps']] == ['Flatten', 'Linear']
