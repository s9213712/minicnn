"""Public API surface for cuda_native."""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import (
    CUDA_NATIVE_CAPABILITIES,
    CUDA_NATIVE_SUPPORT_TIERS,
    get_cuda_native_capabilities,
)
from minicnn.cuda_native.gpu_dispatch import build_gpu_dispatch_plan
from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs
from minicnn.cuda_native.gpu_training_lowering import build_gpu_training_lowering_plan
from minicnn.cuda_native.graph import NativeGraph, build_graph
from minicnn.cuda_native.validators import validate_cuda_native_model_config
from minicnn.model_spec import resolve_model_config

_GPU_NATIVE_SINGLE_CONV_ACTIVATIONS = ('ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh')


_SUPPORTED_DATASET_TYPES = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_datasets']
)
_SUPPORTED_LOSS_TYPES = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_losses']
)
_SUPPORTED_OPTIMIZERS = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_optimizers']
)
_SUPPORTED_EXECUTION_MODES = frozenset(
    str(item) for item in get_cuda_native_capabilities().get('execution_modes_supported', ['reference_numpy'])
)
_PLANNED_EXECUTION_MODES = frozenset(
    str(item) for item in get_cuda_native_capabilities().get('execution_modes_planned', ['gpu_native'])
)
_DEFAULT_EXECUTION_MODE = str(
    get_cuda_native_capabilities().get('default_execution_mode', 'gpu_native_auto')
)
_SUPPORTED_SCHEDULERS = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_schedulers']
)
_SCHEDULER_ALIASES = {
    'step': 'StepLR',
    'steplr': 'StepLR',
    'cosine': 'CosineAnnealingLR',
    'cosineannealinglr': 'CosineAnnealingLR',
    'plateau': 'ReduceLROnPlateau',
    'reducelronplateau': 'ReduceLROnPlateau',
}
_SUPPORT_TIER_ORDER = ('stable', 'beta', 'experimental')


def _assert_execution_mode_invariants() -> None:
    overlap = _SUPPORTED_EXECUTION_MODES & _PLANNED_EXECUTION_MODES
    assert not overlap, f'cuda_native execution mode sets overlap: {sorted(overlap)}'


def assess_cuda_native_execution_readiness(cfg: dict[str, Any]) -> dict[str, object]:
    _assert_execution_mode_invariants()
    caps = get_cuda_native_capabilities()
    readiness_table = dict(caps.get('execution_mode_readiness', {}))
    execution_mode = resolve_cuda_native_execution_mode(cfg)
    selected_mode = str(execution_mode.get('selected_execution_mode', _DEFAULT_EXECUTION_MODE))
    mode_readiness = dict(readiness_table.get(selected_mode, {}))
    model_cfg, _ = _as_mapping('model', cfg.get('model'))
    dataset_cfg, _ = _as_mapping('dataset', cfg.get('dataset'))
    resolved_model_cfg = resolve_model_config(model_cfg)
    layer_types = sorted({
        str(layer.get('type'))
        for layer in resolved_model_cfg.get('layers', [])
        if isinstance(layer, dict) and layer.get('type')
    })
    bootstrap_subset = set(str(item) for item in mode_readiness.get('bootstrap_subset_ops', []))
    kernel_specs = {
        spec.op_name: {
            'forward_status': spec.forward_status,
            'backward_status': spec.backward_status,
            'category': spec.category,
        }
        for spec in list_gpu_kernel_specs()
    }
    supported_ops = sorted(op for op in layer_types if op in bootstrap_subset)
    missing_ops = sorted(op for op in layer_types if op not in bootstrap_subset)
    remaining_blockers = [str(item) for item in mode_readiness.get('remaining_blockers', [])]
    dispatch_plan_summary: dict[str, object] | None = None
    training_lowering_plan_summary: dict[str, object] | None = None
    validation_input_shape = _validation_input_shape(dataset_cfg)
    if validation_input_shape is not None:
        try:
            graph = build_cuda_native_graph(model_cfg, validation_input_shape)
            dispatch_plan = build_gpu_dispatch_plan(graph)
            dispatch_plan_summary = dispatch_plan.summary()
            loss_cfg, _ = _as_mapping('loss', cfg.get('loss'))
            optim_cfg, _ = _as_mapping('optimizer', cfg.get('optimizer'))
            train_cfg, _ = _as_mapping('train', cfg.get('train'))
            training_lowering_plan = build_gpu_training_lowering_plan(
                graph,
                loss_cfg=loss_cfg,
                optim_cfg=optim_cfg,
                train_cfg=train_cfg,
            )
            training_lowering_plan_summary = training_lowering_plan.summary()
            if selected_mode in {'gpu_native', 'gpu_native_auto'}:
                supported_from_plan = {
                    str(step.get('op_name'))
                    for step in dispatch_plan_summary.get('steps', [])
                    if bool(step.get('supported', False))
                }
                supported_ops = sorted(op for op in layer_types if op in supported_from_plan)
                missing_ops = sorted(op for op in layer_types if op not in supported_from_plan)
                for step in dispatch_plan_summary.get('steps', []):
                    if bool(step.get('supported', False)):
                        continue
                    op_name = str(step.get('op_name'))
                    kernel_specs[op_name] = {
                        'forward_status': 'outside_bootstrap',
                        'backward_status': 'outside_bootstrap',
                        'category': 'unsupported',
                    }
        except ValueError:
            dispatch_plan_summary = None
            training_lowering_plan_summary = None
    return {
        'selected_execution_mode': selected_mode,
        'status': str(mode_readiness.get('status', 'unknown')),
        'ready': bool(mode_readiness.get('ready', False)),
        'tensor_execution_device': str(mode_readiness.get('tensor_execution_device', 'unknown')),
        'effective_execution_mode': str(execution_mode.get('effective_execution_mode', selected_mode)),
        'fallback_execution_mode': execution_mode.get('fallback_execution_mode'),
        'fallback_available': bool(execution_mode.get('fallback_available', False)),
        'fallback_active': bool(execution_mode.get('fallback_active', False)),
        'fallback_reason': str(execution_mode.get('fallback_reason', 'not_configured')),
        'requested_ops': layer_types,
        'bootstrap_subset_complete': len(missing_ops) == 0,
        'bootstrap_supported_ops': supported_ops,
        'bootstrap_missing_ops': missing_ops,
        'kernel_readiness_for_requested_ops': {
            op_name: kernel_specs.get(
                op_name,
                {
                    'forward_status': 'outside_bootstrap',
                    'backward_status': 'outside_bootstrap',
                    'category': 'unknown',
                },
            )
            for op_name in layer_types
        },
        'dispatch_plan': dispatch_plan_summary,
        'training_lowering_plan': training_lowering_plan_summary,
        'remaining_blockers': remaining_blockers,
    }


def resolve_cuda_native_execution_mode(cfg: dict[str, Any]) -> dict[str, object]:
    _assert_execution_mode_invariants()
    engine_cfg, _ = _as_mapping('engine', cfg.get('engine'))
    selected_mode = str(engine_cfg.get('execution_mode', _DEFAULT_EXECUTION_MODE) or _DEFAULT_EXECUTION_MODE)
    if selected_mode == 'gpu_native_auto':
        auto_policy = _resolve_gpu_native_auto_policy(cfg)
        effective_mode = 'gpu_native' if bool(auto_policy['gpu_native_ready']) else 'reference_numpy'
        tensor_execution_device = 'gpu' if effective_mode == 'gpu_native' else 'cpu'
        return {
            'execution_mode': effective_mode,
            'selected_execution_mode': selected_mode,
            'effective_execution_mode': effective_mode,
            'tensor_execution_device': tensor_execution_device,
            'tensors_ran_on': tensor_execution_device,
            'gpu_execution': effective_mode == 'gpu_native',
            'planned': False,
            'supported': True,
            'gpu_native_ready': bool(auto_policy['gpu_native_ready']),
            'gpu_native_lowering_ready': bool(auto_policy['gpu_native_lowering_ready']),
            'gpu_native_runtime_ready': bool(auto_policy['gpu_native_runtime_ready']),
            'fallback_execution_mode': 'reference_numpy',
            'fallback_available': True,
            'fallback_active': effective_mode != 'gpu_native',
            'fallback_reason': str(auto_policy['fallback_reason']),
        }
    if selected_mode in _SUPPORTED_EXECUTION_MODES:
        effective_mode = selected_mode
        tensor_execution_device = 'cpu' if selected_mode == 'reference_numpy' else 'gpu'
        return {
            'execution_mode': effective_mode,
            'selected_execution_mode': selected_mode,
            'effective_execution_mode': effective_mode,
            'tensor_execution_device': tensor_execution_device,
            'tensors_ran_on': tensor_execution_device,
            'gpu_execution': tensor_execution_device == 'gpu',
            'planned': False,
            'supported': True,
        }
    if selected_mode in _PLANNED_EXECUTION_MODES:
        return {
            'execution_mode': 'unsupported',
            'selected_execution_mode': selected_mode,
            'effective_execution_mode': 'unsupported',
            'tensor_execution_device': 'gpu',
            'tensors_ran_on': 'gpu',
            'gpu_execution': False,
            'planned': True,
            'supported': False,
        }
    return {
        'execution_mode': 'unsupported',
        'selected_execution_mode': selected_mode,
        'effective_execution_mode': 'unsupported',
        'tensor_execution_device': 'unknown',
        'tensors_ran_on': 'unknown',
        'gpu_execution': False,
        'planned': False,
        'supported': False,
    }


def _cuda_runtime_ready_for_gpu_native() -> tuple[bool, str]:
    try:
        from minicnn.core.cuda_backend import check_cuda_ready

        readiness = check_cuda_ready()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)
    if bool(readiness.get('ready', False)):
        return True, 'not_needed'
    return False, str(readiness.get('error') or 'cuda_runtime_not_ready')


def _resolve_gpu_native_auto_policy(cfg: dict[str, Any]) -> dict[str, object]:
    model_cfg, _ = _as_mapping('model', cfg.get('model'))
    dataset_cfg, _ = _as_mapping('dataset', cfg.get('dataset'))
    loss_cfg, _ = _as_mapping('loss', cfg.get('loss'))
    optim_cfg, _ = _as_mapping('optimizer', cfg.get('optimizer'))
    train_cfg, _ = _as_mapping('train', cfg.get('train'))
    validation_input_shape = _validation_input_shape(dataset_cfg)
    lowering_ready = False
    lowering_reason = 'validation_input_shape_unavailable'
    if validation_input_shape is not None:
        try:
            graph = build_cuda_native_graph(model_cfg, validation_input_shape)
            plan = build_gpu_training_lowering_plan(
                graph,
                loss_cfg=loss_cfg,
                optim_cfg=optim_cfg,
                train_cfg=train_cfg,
            )
            policy = plan.fallback_policy()
            lowering_ready = bool(policy.get('gpu_native_ready', False))
            lowering_reason = str(policy.get('fallback_reason', 'gpu_native_training_lowering_not_ready'))
        except ValueError as exc:
            lowering_reason = str(exc)
    runtime_ready, runtime_reason = _cuda_runtime_ready_for_gpu_native()
    gpu_native_ready = lowering_ready and runtime_ready
    if gpu_native_ready:
        fallback_reason = 'not_needed'
    elif not lowering_ready:
        fallback_reason = lowering_reason
    else:
        fallback_reason = runtime_reason
    return {
        'gpu_native_ready': gpu_native_ready,
        'gpu_native_lowering_ready': lowering_ready,
        'gpu_native_runtime_ready': runtime_ready,
        'fallback_reason': fallback_reason,
    }


def _validate_engine_cfg(engine_cfg: dict[str, Any]) -> list[str]:
    _assert_execution_mode_invariants()
    errors: list[str] = []
    execution_mode = str(engine_cfg.get('execution_mode', _DEFAULT_EXECUTION_MODE) or _DEFAULT_EXECUTION_MODE)
    if execution_mode in _SUPPORTED_EXECUTION_MODES:
        return errors
    if execution_mode in _PLANNED_EXECUTION_MODES:
        return errors
    supported = ', '.join(sorted(_SUPPORTED_EXECUTION_MODES | _PLANNED_EXECUTION_MODES))
    errors.append(
        f'cuda_native does not recognize engine.execution_mode={execution_mode!r}. Supported/planned: {supported}.'
    )
    return errors


def _validate_gpu_native_training_subset(
    cfg: dict[str, Any],
    graph: NativeGraph,
    *,
    loss_cfg: dict[str, Any],
    optim_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    nodes = list(graph.topological_order())
    ops = [node.op_type for node in nodes]
    allowed_ops = {
        ('Linear',),
        ('Flatten', 'Linear'),
        ('Linear', 'ReLU', 'Linear'),
        ('Flatten', 'Linear', 'ReLU', 'Linear'),
        ('Linear', 'LeakyReLU', 'Linear'),
        ('Flatten', 'Linear', 'LeakyReLU', 'Linear'),
        ('Linear', 'GELU', 'Linear'),
        ('Flatten', 'Linear', 'GELU', 'Linear'),
        ('Linear', 'SiLU', 'Linear'),
        ('Flatten', 'Linear', 'SiLU', 'Linear'),
        ('Linear', 'Sigmoid', 'Linear'),
        ('Flatten', 'Linear', 'Sigmoid', 'Linear'),
        ('Linear', 'Tanh', 'Linear'),
        ('Flatten', 'Linear', 'Tanh', 'Linear'),
        ('MaxPool2d', 'Flatten', 'Linear'),
        ('AvgPool2d', 'Flatten', 'Linear'),
        ('BatchNorm2d', 'Flatten', 'Linear'),
        ('LayerNorm2d', 'Flatten', 'Linear'),
        ('GroupNorm', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'GELU', 'PointwiseConv2d', 'Flatten', 'Linear'),
        ('GlobalAvgPool2d', 'Flatten', 'Linear'),
        ('AdaptiveAvgPool2d', 'Flatten', 'Linear'),
        ('Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'),
    }
    for _conv_op in ('Conv2d', 'PointwiseConv2d', 'DepthwiseConv2d'):
        allowed_ops.add((_conv_op, 'Flatten', 'Linear'))
        for _activation in _GPU_NATIVE_SINGLE_CONV_ACTIVATIONS:
            allowed_ops.add((_conv_op, _activation, 'Flatten', 'Linear'))
            if _conv_op != 'PointwiseConv2d':
                allowed_ops.add((_conv_op, _activation, 'MaxPool2d', 'Flatten', 'Linear'))
        if _conv_op != 'PointwiseConv2d':
            allowed_ops.add((_conv_op, 'MaxPool2d', 'Flatten', 'Linear'))
    for _activation in _GPU_NATIVE_SINGLE_CONV_ACTIVATIONS:
        allowed_ops.add(('Conv2d', _activation, 'Conv2d', _activation, 'MaxPool2d', 'Flatten', 'Linear'))
    if tuple(ops) not in allowed_ops:
        errors.append(
            'cuda_native gpu_native train-native currently supports only the narrow '
            'Linear training subset ops=[Linear], [Flatten, Linear], '
            '[Linear, ReLU, Linear], [Flatten, Linear, ReLU, Linear], '
            '[Linear, LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Linear], '
            '[Flatten, Linear, LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Linear], '
            '[MaxPool2d, Flatten, Linear], [AvgPool2d, Flatten, Linear], '
            '[BatchNorm2d, Flatten, Linear], [LayerNorm2d, Flatten, Linear], '
            '[GroupNorm, Flatten, Linear], '
            '[DepthwiseConv2d, LayerNorm2d, Flatten, Linear], '
            '[DepthwiseConv2d, LayerNorm2d, PointwiseConv2d, Flatten, Linear], '
            '[DepthwiseConv2d, LayerNorm2d, PointwiseConv2d, GELU, PointwiseConv2d, Flatten, Linear], '
            '[GlobalAvgPool2d, Flatten, Linear], '
            '[AdaptiveAvgPool2d, Flatten, Linear], [Conv2d, Flatten, Linear], '
            '[Conv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Flatten, Linear], '
            '[PointwiseConv2d, Flatten, Linear], '
            '[PointwiseConv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Flatten, Linear], '
            '[DepthwiseConv2d, Flatten, Linear], '
            '[DepthwiseConv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Flatten, Linear], '
            '[DepthwiseConv2d, MaxPool2d, Flatten, Linear], '
            '[DepthwiseConv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, MaxPool2d, Flatten, Linear], '
            '[Conv2d, MaxPool2d, Flatten, Linear], '
            '[Conv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, MaxPool2d, Flatten, Linear], or '
            '[Conv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Conv2d, same activation, MaxPool2d, Flatten, Linear], '
            f'got {ops}.'
        )
    conv_constraint_ops = {
        ('DepthwiseConv2d', 'LayerNorm2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'GELU', 'PointwiseConv2d', 'Flatten', 'Linear'),
        ('Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'),
    }
    for _conv_op in ('Conv2d', 'PointwiseConv2d', 'DepthwiseConv2d'):
        conv_constraint_ops.add((_conv_op, 'Flatten', 'Linear'))
        for _activation in _GPU_NATIVE_SINGLE_CONV_ACTIVATIONS:
            conv_constraint_ops.add((_conv_op, _activation, 'Flatten', 'Linear'))
            if _conv_op != 'PointwiseConv2d':
                conv_constraint_ops.add((_conv_op, _activation, 'MaxPool2d', 'Flatten', 'Linear'))
        if _conv_op != 'PointwiseConv2d':
            conv_constraint_ops.add((_conv_op, 'MaxPool2d', 'Flatten', 'Linear'))
    for _activation in _GPU_NATIVE_SINGLE_CONV_ACTIVATIONS:
        conv_constraint_ops.add(('Conv2d', _activation, 'Conv2d', _activation, 'MaxPool2d', 'Flatten', 'Linear'))
    if tuple(ops) in conv_constraint_ops:
        conv_attr_nodes = [nodes[0]]
        if (
            len(ops) == 7
            and ops[0] == 'Conv2d'
            and ops[1] in _GPU_NATIVE_SINGLE_CONV_ACTIVATIONS
            and ops[2] == 'Conv2d'
            and ops[3] == ops[1]
            and ops[4:] == ['MaxPool2d', 'Flatten', 'Linear']
        ):
            conv_attr_nodes.append(nodes[2])
        if ops == ['DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear']:
            conv_attr_nodes.append(nodes[2])
        if ops == ['DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'GELU', 'PointwiseConv2d', 'Flatten', 'Linear']:
            conv_attr_nodes.extend((nodes[2], nodes[4]))

        def _pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        for conv_node in conv_attr_nodes:
            conv_attrs = dict(getattr(conv_node, 'attrs', {}) or {})
            conv_op = str(getattr(conv_node, 'op_type', 'Conv2d'))
            if bool(conv_attrs.get('bias', False)):
                errors.append(f'cuda_native gpu_native {conv_op} train-native subset currently requires bias=false.')
            if conv_op != 'DepthwiseConv2d' and int(conv_attrs.get('groups', 1)) != 1:
                errors.append(f'cuda_native gpu_native {conv_op} train-native subset currently requires groups=1.')
            if _pair(conv_attrs.get('stride', 1), 1) != (1, 1):
                errors.append(f'cuda_native gpu_native {conv_op} train-native subset currently requires stride=1.')
            if _pair(conv_attrs.get('padding', 0), 0) != (0, 0):
                errors.append(f'cuda_native gpu_native {conv_op} train-native subset currently requires padding=0.')
            if _pair(conv_attrs.get('dilation', 1), 1) != (1, 1):
                errors.append(f'cuda_native gpu_native {conv_op} train-native subset currently requires dilation=1.')
    if ops == ['AvgPool2d', 'Flatten', 'Linear']:
        pool_attrs = dict(getattr(nodes[0], 'attrs', {}) or {})

        def _pool_pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        if _pool_pair(pool_attrs.get('kernel_size', 2), 2) != (2, 2):
            errors.append('cuda_native gpu_native AvgPool2d train-native subset currently requires kernel_size=2.')
        if _pool_pair(pool_attrs.get('stride', pool_attrs.get('kernel_size', 2)), 2) != (2, 2):
            errors.append('cuda_native gpu_native AvgPool2d train-native subset currently requires stride=2.')
        if _pool_pair(pool_attrs.get('padding', 0), 0) != (0, 0):
            errors.append('cuda_native gpu_native AvgPool2d train-native subset currently requires padding=0.')
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    if loss_type != 'CrossEntropyLoss' and ops not in (['Linear'], ['Flatten', 'Linear']):
        errors.append('cuda_native gpu_native train-native currently supports MSELoss/BCEWithLogitsLoss only for the Linear subset.')
    optimizer_type = str(optim_cfg.get('type', 'SGD')).lower()
    if ops in (['Linear'], ['Flatten', 'Linear']):
        if optimizer_type not in {'sgd', 'adam', 'adamw', 'rmsprop'}:
            errors.append('cuda_native gpu_native Linear train-native currently supports optimizer.type in {SGD, Adam, AdamW, RMSprop}.')
        if optimizer_type == 'adam' and float(optim_cfg.get('weight_decay', 0.0)) != 0.0:
            errors.append('cuda_native gpu_native Linear train-native currently requires Adam weight_decay=0.0; use AdamW for decoupled weight decay.')
    else:
        if optimizer_type != 'sgd':
            errors.append('cuda_native gpu_native non-Linear train-native currently supports only optimizer.type=SGD.')
    if bool(train_cfg.get('amp', False)):
        errors.append('cuda_native gpu_native train-native currently requires train.amp=false.')
    return errors


def _gpu_native_bootstrap_error(readiness: dict[str, object]) -> str:
    supported_ops = list(readiness.get('bootstrap_supported_ops', []))
    missing_ops = list(readiness.get('bootstrap_missing_ops', []))
    blockers = list(readiness.get('remaining_blockers', []))
    if missing_ops:
        return (
            'cuda_native gpu_native bootstrap subset coverage: '
            f'supported={supported_ops}, outside_bootstrap={missing_ops}, blockers={blockers}.'
        )
    return (
        'cuda_native gpu_native bootstrap subset coverage: '
        f'all requested ops are within bootstrap subset={supported_ops}, blockers={blockers}.'
    )


def _matched_support_tiers(items: set[str], bucket: str) -> dict[str, list[str]]:
    return {
        tier: sorted(items & set(values.get(bucket, [])))
        for tier, values in CUDA_NATIVE_SUPPORT_TIERS.items()
    }


def _highest_tier_from_matches(matches: dict[str, list[str]]) -> str:
    highest = 'stable'
    for tier in _SUPPORT_TIER_ORDER:
        if matches.get(tier):
            highest = tier
    return highest


def assess_cuda_native_support_tier(cfg: dict[str, Any]) -> dict[str, object]:
    model_cfg, _ = _as_mapping('model', cfg.get('model'))
    optimizer_cfg, _ = _as_mapping('optimizer', cfg.get('optimizer'))
    loss_cfg, _ = _as_mapping('loss', cfg.get('loss'))
    train_cfg, _ = _as_mapping('train', cfg.get('train'))

    resolved_model_cfg = resolve_model_config(model_cfg)
    layer_types = {
        str(layer.get('type'))
        for layer in resolved_model_cfg.get('layers', [])
        if isinstance(layer, dict) and layer.get('type')
    }
    optimizer_types = {str(optimizer_cfg.get('type', 'SGD'))}
    loss_types = {str(loss_cfg.get('type', 'CrossEntropyLoss'))}
    features: set[str] = set()
    if bool(train_cfg.get('amp', False)):
        features.add('amp')
    if any(isinstance(layer, dict) and layer.get('output') for layer in resolved_model_cfg.get('layers', [])):
        features.add('ordered_dag')
    if any(
        isinstance(layer, dict) and (
            str(layer.get('type')) in {'Add', 'Concat'}
            or (isinstance(layer.get('inputs'), list) and len(layer.get('inputs')) > 1)
        )
        for layer in resolved_model_cfg.get('layers', [])
    ):
        features.add('branching_graph')
    matches = {
        'ops': _matched_support_tiers(layer_types, 'ops'),
        'optimizers': _matched_support_tiers(optimizer_types, 'optimizers'),
        'losses': _matched_support_tiers(loss_types, 'losses'),
        'features': _matched_support_tiers(features, 'features'),
    }
    highest_tier = 'stable'
    for bucket_matches in matches.values():
        bucket_highest = _highest_tier_from_matches(bucket_matches)
        if _SUPPORT_TIER_ORDER.index(bucket_highest) > _SUPPORT_TIER_ORDER.index(highest_tier):
            highest_tier = bucket_highest
    return {
        'highest_tier': highest_tier,
        'ops_by_tier': matches['ops'],
        'optimizers_by_tier': matches['optimizers'],
        'losses_by_tier': matches['losses'],
        'features_by_tier': matches['features'],
    }


def _as_mapping(name: str, value: Any) -> tuple[dict[str, Any], list[str]]:
    if value is None:
        return {}, []
    if isinstance(value, dict):
        return value, []
    return {}, [f'{name} must be a mapping']


def _coerce_float(name: str, value: Any) -> tuple[float | None, list[str]]:
    try:
        return float(value), []
    except (TypeError, ValueError):
        return None, [f'{name} must be numeric, got {value!r}']


def _coerce_int(name: str, value: Any) -> tuple[int | None, list[str]]:
    try:
        return int(value), []
    except (TypeError, ValueError):
        return None, [f'{name} must be an integer, got {value!r}']


def _validate_dataset_cfg(dataset_cfg: dict[str, Any]) -> list[str]:
    dtype = str(dataset_cfg.get('type', 'random'))
    if dtype not in _SUPPORTED_DATASET_TYPES:
        supported = ', '.join(sorted(_SUPPORTED_DATASET_TYPES))
        return [
            f'cuda_native does not support dataset.type={dtype!r}. Supported: {supported}.'
        ]
    return []


def _validate_loss_cfg(loss_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    if loss_type not in _SUPPORTED_LOSS_TYPES:
        supported = ', '.join(sorted(_SUPPORTED_LOSS_TYPES))
        return [
            f'cuda_native does not support loss.type={loss_type!r}. Supported: {supported}.'
        ]
    if 'label_smoothing' in loss_cfg:
        smoothing, smoothing_errors = _coerce_float(
            'loss.label_smoothing',
            loss_cfg.get('label_smoothing', 0.0),
        )
        errors.extend(smoothing_errors)
        if smoothing is not None and not (0.0 <= smoothing < 1.0):
            errors.append('loss.label_smoothing must be in [0, 1) for cuda_native.')
        if smoothing is not None and smoothing > 0.0 and loss_type != 'CrossEntropyLoss':
            errors.append('cuda_native only supports loss.label_smoothing with CrossEntropyLoss.')
    return errors


def _validate_optimizer_cfg(optim_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    opt_type = str(optim_cfg.get('type', 'SGD'))
    if opt_type not in _SUPPORTED_OPTIMIZERS:
        supported = ', '.join(sorted(_SUPPORTED_OPTIMIZERS))
        errors.append(
            f'cuda_native only supports optimizer.type in {{{supported}}}; got {opt_type!r}.'
        )
        return errors

    base_lr, lr_errors = _coerce_float('optimizer.lr', optim_cfg.get('lr', 0.01))
    errors.extend(lr_errors)

    if opt_type == 'SGD':
        momentum_val, momentum_errors = _coerce_float(
            'optimizer.momentum',
            optim_cfg.get('momentum', 0.0),
        )
        errors.extend(momentum_errors)
        if momentum_val is not None and momentum_val < 0.0:
            errors.append('optimizer.momentum must be >= 0 for cuda_native.')
    elif opt_type in {'Adam', 'AdamW'}:
        for field in ('beta1', 'beta2', 'eps'):
            if field not in optim_cfg:
                continue
            value, value_errors = _coerce_float(f'optimizer.{field}', optim_cfg[field])
            errors.extend(value_errors)
            if value is None:
                continue
            if field in {'beta1', 'beta2'} and not (0.0 <= value < 1.0):
                errors.append(f'optimizer.{field} must be in [0, 1) for {opt_type}.')
            if field == 'eps' and value <= 0.0:
                errors.append(f'optimizer.eps must be > 0 for {opt_type}.')
    elif opt_type == 'RMSprop':
        alpha_val, alpha_errors = _coerce_float(
            'optimizer.alpha',
            optim_cfg.get('alpha', 0.99),
        )
        errors.extend(alpha_errors)
        if alpha_val is not None and not (0.0 <= alpha_val < 1.0):
            errors.append('optimizer.alpha must be in [0, 1) for RMSprop.')
        eps_val, eps_errors = _coerce_float(
            'optimizer.eps',
            optim_cfg.get('eps', 1e-8),
        )
        errors.extend(eps_errors)
        if eps_val is not None and eps_val <= 0.0:
            errors.append('optimizer.eps must be > 0 for RMSprop.')
        momentum_val, momentum_errors = _coerce_float(
            'optimizer.momentum',
            optim_cfg.get('momentum', 0.0),
        )
        errors.extend(momentum_errors)
        if momentum_val is not None and momentum_val < 0.0:
            errors.append('optimizer.momentum must be >= 0 for RMSprop.')

    grad_clip_val, grad_clip_errors = _coerce_float(
        'optimizer.grad_clip_global',
        optim_cfg.get('grad_clip_global', 0.0),
    )
    errors.extend(grad_clip_errors)
    if grad_clip_val is not None and grad_clip_val < 0.0:
        errors.append('optimizer.grad_clip_global must be >= 0 for cuda_native.')

    for field in ('lr_conv1', 'lr_conv', 'lr_fc'):
        if field not in optim_cfg:
            continue
        field_lr, field_errors = _coerce_float(f'optimizer.{field}', optim_cfg[field])
        errors.extend(field_errors)
        if base_lr is not None and field_lr is not None and field_lr != base_lr:
            errors.append(
                f'cuda_native uses a single optimizer.lr; optimizer.{field}={field_lr} '
                f'must match optimizer.lr={base_lr}.'
            )
    return errors


def _validate_scheduler_cfg(scheduler_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    enabled = bool(scheduler_cfg.get('enabled', False))
    scheduler_type = str(scheduler_cfg.get('type', 'none') or 'none')
    normalized = scheduler_type.lower()
    no_scheduler = normalized in {'none', 'disabled', 'null', ''}
    if not enabled:
        return errors
    if no_scheduler:
        errors.append('cuda_native expects scheduler.enabled=false when no scheduler is used.')
        return errors

    canonical = _SCHEDULER_ALIASES.get(normalized, scheduler_type)
    if canonical not in _SUPPORTED_SCHEDULERS:
        supported = ', '.join(sorted(_SUPPORTED_SCHEDULERS))
        errors.append(
            f'cuda_native does not support scheduler.type={scheduler_type!r}. '
            f'Supported: {supported}.'
        )
        return errors

    if canonical == 'StepLR':
        step_size, step_size_errors = _coerce_int(
            'scheduler.step_size',
            scheduler_cfg.get('step_size', 10),
        )
        gamma, gamma_errors = _coerce_float(
            'scheduler.gamma',
            scheduler_cfg.get('gamma', 0.5),
        )
        min_lr, min_lr_errors = _coerce_float(
            'scheduler.min_lr',
            scheduler_cfg.get('min_lr', 0.0),
        )
        errors.extend(step_size_errors)
        errors.extend(gamma_errors)
        errors.extend(min_lr_errors)
        if step_size is not None and step_size <= 0:
            errors.append('scheduler.step_size must be > 0 for StepLR.')
        if gamma is not None and gamma < 0.0:
            errors.append('scheduler.gamma must be >= 0 for StepLR.')
        if min_lr is not None and min_lr < 0.0:
            errors.append('scheduler.min_lr must be >= 0 for StepLR.')
    elif canonical == 'CosineAnnealingLR':
        t_max, t_max_errors = _coerce_int(
            'scheduler.T_max',
            scheduler_cfg.get('T_max', scheduler_cfg.get('t_max', 10)),
        )
        lr_min, lr_min_errors = _coerce_float(
            'scheduler.lr_min',
            scheduler_cfg.get('lr_min', 0.0),
        )
        errors.extend(t_max_errors)
        errors.extend(lr_min_errors)
        if t_max is not None and t_max <= 0:
            errors.append('scheduler.T_max must be > 0 for CosineAnnealingLR.')
        if lr_min is not None and lr_min < 0.0:
            errors.append('scheduler.lr_min must be >= 0 for CosineAnnealingLR.')
    elif canonical == 'ReduceLROnPlateau':
        factor, factor_errors = _coerce_float(
            'scheduler.factor',
            scheduler_cfg.get('factor', 0.5),
        )
        patience, patience_errors = _coerce_int(
            'scheduler.patience',
            scheduler_cfg.get('patience', 3),
        )
        min_lr, min_lr_errors = _coerce_float(
            'scheduler.min_lr',
            scheduler_cfg.get('min_lr', 1e-5),
        )
        errors.extend(factor_errors)
        errors.extend(patience_errors)
        errors.extend(min_lr_errors)
        if factor is not None and not (0.0 < factor <= 1.0):
            errors.append('scheduler.factor must be in (0, 1] for ReduceLROnPlateau.')
        if patience is not None and patience < 0:
            errors.append('scheduler.patience must be >= 0 for ReduceLROnPlateau.')
        if min_lr is not None and min_lr < 0.0:
            errors.append('scheduler.min_lr must be >= 0 for ReduceLROnPlateau.')
    return errors


def _validate_train_cfg(
    train_cfg: dict[str, Any],
    *,
    execution_mode: str = _DEFAULT_EXECUTION_MODE,
) -> list[str]:
    errors: list[str] = []
    grad_accum_steps, grad_accum_errors = _coerce_int(
        'train.grad_accum_steps',
        train_cfg.get('grad_accum_steps', 1),
    )
    errors.extend(grad_accum_errors)
    if grad_accum_steps is not None and grad_accum_steps <= 0:
        errors.append('train.grad_accum_steps must be >= 1 for cuda_native.')
    amp_loss_scale, amp_loss_scale_errors = _coerce_float(
        'train.amp_loss_scale',
        train_cfg.get('amp_loss_scale', 128.0),
    )
    errors.extend(amp_loss_scale_errors)
    if amp_loss_scale is not None and amp_loss_scale <= 0.0:
        errors.append('train.amp_loss_scale must be > 0 for cuda_native AMP.')
    amp_scale_growth, amp_scale_growth_errors = _coerce_float(
        'train.amp_scale_growth',
        train_cfg.get('amp_scale_growth', 2.0),
    )
    errors.extend(amp_scale_growth_errors)
    if amp_scale_growth is not None and amp_scale_growth <= 1.0:
        errors.append('train.amp_scale_growth must be > 1 for cuda_native AMP.')
    amp_scale_backoff, amp_scale_backoff_errors = _coerce_float(
        'train.amp_scale_backoff',
        train_cfg.get('amp_scale_backoff', 0.5),
    )
    errors.extend(amp_scale_backoff_errors)
    if amp_scale_backoff is not None and not (0.0 < amp_scale_backoff < 1.0):
        errors.append('train.amp_scale_backoff must be in (0, 1) for cuda_native AMP.')
    amp_scale_window, amp_scale_window_errors = _coerce_int(
        'train.amp_scale_window',
        train_cfg.get('amp_scale_window', 200),
    )
    errors.extend(amp_scale_window_errors)
    if amp_scale_window is not None and amp_scale_window <= 0:
        errors.append('train.amp_scale_window must be >= 1 for cuda_native AMP.')
    device = train_cfg.get('device')
    allowed_devices = {None, 'auto', 'cpu'}
    if execution_mode in {'gpu_native', 'gpu_native_auto'}:
        allowed_devices = {None, 'auto', 'cpu', 'cuda', 'gpu'}
    if device not in allowed_devices:
        errors.append(
            f'cuda_native execution_mode={execution_mode!r} does not accept train.device={device!r}; '
            f'use one of {sorted(str(item) for item in allowed_devices if item is not None)}.'
        )
    return errors


def _validation_input_shape(dataset_cfg: dict[str, Any]) -> tuple[int, ...] | None:
    input_shape = dataset_cfg.get('input_shape')
    if isinstance(input_shape, (list, tuple)) and input_shape:
        return (1, *tuple(int(dim) for dim in input_shape))
    dtype = str(dataset_cfg.get('type', 'random'))
    if dtype == 'cifar10':
        return (1, 3, 32, 32)
    if dtype == 'mnist':
        return (1, 1, 28, 28)
    return None


def validate_cuda_native_config(cfg: dict[str, Any]) -> list[str]:
    """Validate a full experiment config for cuda_native compatibility.

    Returns a list of error strings (empty list = valid).
    """
    errors: list[str] = []

    model_cfg, model_cfg_errors = _as_mapping('model', cfg.get('model'))
    dataset_cfg, dataset_cfg_errors = _as_mapping('dataset', cfg.get('dataset'))
    loss_cfg, loss_cfg_errors = _as_mapping('loss', cfg.get('loss'))
    optim_cfg, optim_cfg_errors = _as_mapping('optimizer', cfg.get('optimizer'))
    scheduler_cfg, scheduler_cfg_errors = _as_mapping('scheduler', cfg.get('scheduler'))
    train_cfg, train_cfg_errors = _as_mapping('train', cfg.get('train'))
    engine_cfg, engine_cfg_errors = _as_mapping('engine', cfg.get('engine'))

    errors.extend(model_cfg_errors)
    errors.extend(dataset_cfg_errors)
    errors.extend(loss_cfg_errors)
    errors.extend(optim_cfg_errors)
    errors.extend(scheduler_cfg_errors)
    errors.extend(train_cfg_errors)
    errors.extend(engine_cfg_errors)

    model_errors = validate_cuda_native_model_config(model_cfg)
    errors.extend(model_errors)
    errors.extend(_validate_engine_cfg(engine_cfg))
    errors.extend(_validate_dataset_cfg(dataset_cfg))
    errors.extend(_validate_loss_cfg(loss_cfg))
    errors.extend(_validate_optimizer_cfg(optim_cfg))
    errors.extend(_validate_scheduler_cfg(scheduler_cfg))
    execution_mode = resolve_cuda_native_execution_mode(cfg)
    selected_execution_mode = str(execution_mode.get('selected_execution_mode', _DEFAULT_EXECUTION_MODE))
    errors.extend(_validate_train_cfg(train_cfg, execution_mode=selected_execution_mode))

    validation_input_shape = _validation_input_shape(dataset_cfg)
    if validation_input_shape is not None and not model_errors:
        try:
            graph = build_cuda_native_graph(model_cfg, validation_input_shape)
            if str(execution_mode.get('selected_execution_mode')) == 'gpu_native':
                errors.extend(
                    _validate_gpu_native_training_subset(
                        cfg,
                        graph,
                        loss_cfg=loss_cfg,
                        optim_cfg=optim_cfg,
                        train_cfg=train_cfg,
                    )
                )
            loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
            if loss_type == 'BCEWithLogitsLoss':
                output_shape = tuple(graph.output_spec.shape) if graph.output_spec is not None else ()
                if len(output_shape) != 2 or int(output_shape[1]) != 1:
                    errors.append(
                        'cuda_native requires model output shape (N, 1) for BCEWithLogitsLoss.'
                    )
        except ValueError as exc:
            message = str(exc)
            if message.startswith('cuda_native validation failed:\n- '):
                errors.extend(message.removeprefix('cuda_native validation failed:\n- ').split('\n- '))
            else:
                errors.append(message)

    return errors


def build_cuda_native_graph(
    model_cfg: dict[str, Any],
    input_shape: tuple[int, ...],
) -> NativeGraph:
    """Build and return a NativeGraph from a model config dict.

    Args:
        model_cfg:   dict with a 'layers' key (same format as flex/autograd).
        input_shape: fixed input shape, e.g. (1, 3, 32, 32).

    Raises:
        ValueError: if the config references unsupported ops or has bad attrs/shapes.
    """
    resolved_model_cfg = resolve_model_config(model_cfg)
    layers = resolved_model_cfg.get('layers', [])
    return build_graph(layers, input_shape)


def get_capability_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_cuda_native_capabilities()
