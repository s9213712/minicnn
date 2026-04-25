from __future__ import annotations


def test_gpu_training_launch_trace_records_ordered_per_phase_lowering_packets():
    from minicnn.cuda_native.api import build_cuda_native_graph
    from minicnn.cuda_native.gpu_training_lowering import (
        build_gpu_training_launch_trace,
        build_gpu_training_lowering_plan,
    )

    graph = build_cuda_native_graph({
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }, (1, 1, 4, 4))
    plan = build_gpu_training_lowering_plan(
        graph,
        loss_cfg={'type': 'CrossEntropyLoss'},
        optim_cfg={'type': 'SGD'},
    )
    trace = [packet.summary() for packet in build_gpu_training_launch_trace(plan)]

    assert [(packet['phase'], packet['lowering_kind']) for packet in trace] == [
        ('forward', 'shape_flatten_shim'),
        ('forward', 'linear_affine_shim'),
        ('loss', 'softmax_xent_grad_loss_acc'),
        ('backward', 'dense_backward_full'),
        ('optimizer', 'apply_sgd_update'),
    ]
    assert [packet['ordinal'] for packet in trace] == [0, 1, 2, 3, 4]
    assert trace[0]['required_symbols'] == []
    assert trace[1]['required_symbols'] == ['dense_forward']
    assert trace[3]['required_symbols'] == ['dense_backward_full']
    assert trace[4]['param_keys'] == ['_w_linear_1', '_b_linear_1']
    assert plan.summary()['training_launch_trace'] == trace


def test_gpu_training_per_op_lowering_manifest_exposes_helper_transition_contract():
    from minicnn.cuda_native.api import build_cuda_native_graph
    from minicnn.cuda_native.gpu_training_lowering import build_gpu_training_lowering_plan

    graph = build_cuda_native_graph({
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }, (1, 1, 4, 4))
    plan = build_gpu_training_lowering_plan(
        graph,
        loss_cfg={'type': 'CrossEntropyLoss'},
        optim_cfg={'type': 'SGD'},
    )
    manifest = plan.summary()['per_op_lowering_shim']

    assert manifest['schema_version'] == 1
    assert manifest['manifest_kind'] == 'cuda_native_gpu_training_per_op_lowering_shim'
    assert manifest['ready'] is True
    assert manifest['helper_backed'] is True
    assert manifest['helper'] == 'native_gpu_linear_training_step'
    assert manifest['transition_policy'] == 'helper_backed_until_runtime_per_op_executor'
    assert manifest['launch_count'] == 5
    assert manifest['fallback_policy']['fallback_active'] is False
    assert manifest['required_symbols_by_phase'] == {
        'backward': ['dense_backward_full'],
        'forward': ['dense_forward'],
        'loss': ['softmax_xent_grad_loss_acc'],
        'optimizer': ['apply_sgd_update'],
    }
    assert [packet['depends_on_ordinals'] for packet in manifest['packets']] == [
        [],
        [0],
        [1],
        [2],
        [3],
    ]
    assert [(packet['phase'], packet['lowering_kind']) for packet in manifest['packets']] == [
        ('forward', 'shape_flatten_shim'),
        ('forward', 'linear_affine_shim'),
        ('loss', 'softmax_xent_grad_loss_acc'),
        ('backward', 'dense_backward_full'),
        ('optimizer', 'apply_sgd_update'),
    ]
