from __future__ import annotations


def test_gpu_training_lowering_summary_keeps_numpy_fallback_for_ready_gpu_subset():
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
    policy = plan.summary()['fallback_policy']

    assert plan.ready is True
    assert policy['selected_execution_mode'] == 'gpu_native'
    assert policy['gpu_native_ready'] is True
    assert policy['fallback_execution_mode'] == 'reference_numpy'
    assert policy['fallback_available'] is True
    assert policy['fallback_active'] is False
    assert policy['fallback_reason'] == 'not_needed'


def test_gpu_training_lowering_summary_activates_numpy_fallback_for_unsupported_gpu_subset():
    from minicnn.cuda_native.api import build_cuda_native_graph
    from minicnn.cuda_native.gpu_training_lowering import build_gpu_training_lowering_plan

    graph = build_cuda_native_graph({
        'layers': [
            {'type': 'Dropout', 'p': 0.5},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }, (1, 1, 4, 4))
    plan = build_gpu_training_lowering_plan(
        graph,
        loss_cfg={'type': 'CrossEntropyLoss'},
        optim_cfg={'type': 'SGD'},
    )
    policy = plan.summary()['fallback_policy']

    assert plan.ready is False
    assert policy['selected_execution_mode'] == 'gpu_native'
    assert policy['gpu_native_ready'] is False
    assert policy['fallback_execution_mode'] == 'reference_numpy'
    assert policy['fallback_available'] is True
    assert policy['fallback_active'] is True
    assert 'unsupported gpu_native training subset' in policy['fallback_reason']


def test_gpu_training_lowering_uses_numpy_fallback_when_conv_helper_constraints_do_not_hold():
    from minicnn.cuda_native.api import build_cuda_native_graph
    from minicnn.cuda_native.gpu_training_lowering import build_gpu_training_lowering_plan

    graph = build_cuda_native_graph({
        'layers': [
            {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3, 'padding': 1, 'bias': True},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ],
    }, (1, 1, 4, 4))
    plan = build_gpu_training_lowering_plan(
        graph,
        loss_cfg={'type': 'CrossEntropyLoss'},
        optim_cfg={'type': 'SGD'},
    )
    policy = plan.summary()['fallback_policy']

    assert plan.ready is False
    assert policy['fallback_active'] is True
    assert 'bias=false' in policy['fallback_reason']
    assert 'padding=0' in policy['fallback_reason']
