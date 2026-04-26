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


def test_gpu_training_lowering_plans_generic_per_op_trace_for_mlp_runtime_executor():
    from minicnn.cuda_native.api import build_cuda_native_graph
    from minicnn.cuda_native.gpu_training_lowering import build_gpu_training_lowering_plan

    graph = build_cuda_native_graph({
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 4},
            {'type': 'ReLU'},
            {'type': 'Linear', 'out_features': 4},
            {'type': 'GELU'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }, (1, 1, 4, 4))
    plan = build_gpu_training_lowering_plan(
        graph,
        loss_cfg={'type': 'CrossEntropyLoss'},
        optim_cfg={'type': 'SGD'},
    )
    manifest = plan.summary()['per_op_lowering_shim']

    assert plan.ready is True
    assert plan.subset_name == 'generic_mlp'
    assert plan.helper is None
    assert manifest['helper_backed'] is False
    assert manifest['graph_wide_lowering_ready'] is True
    assert manifest['runtime_executor_ready'] is True
    assert manifest['transition_policy'] == 'runtime_per_op_executor'
    assert plan.unsupported_reasons == ()
    assert [(step.phase, step.lowering_kind) for step in plan.backward_steps] == [
        ('backward', 'dense_backward_full'),
        ('backward', 'gelu_backward'),
        ('backward', 'dense_backward_full'),
        ('backward', 'apply_relu_backward'),
        ('backward', 'dense_backward_full'),
        ('backward', 'shape_flatten_backward'),
    ]
    assert plan.optimizer_steps[-1].param_keys == (
        '_w_linear_1',
        '_b_linear_1',
        '_w_linear_3',
        '_b_linear_3',
        '_w_linear_5',
        '_b_linear_5',
    )
    assert manifest['fallback_policy']['fallback_active'] is False
