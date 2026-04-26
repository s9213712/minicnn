from __future__ import annotations

from typing import Any

from minicnn.cuda_native.graph import NativeGraph
from minicnn.unified._cuda_native_context import NativeTrainingContext

_SINGLE_STAGE_ACTIVATIONS = {'GELU', 'LeakyReLU', 'ReLU', 'SiLU', 'Sigmoid', 'Tanh'}


def _is_generic_mlp_ops(ops: list[str]) -> bool:
    remaining = ops[1:] if ops and ops[0] == 'Flatten' else list(ops)
    if len(remaining) < 5 or remaining[0] != 'Linear' or remaining[-1] != 'Linear':
        return False
    expect_activation = True
    for op in remaining[1:-1]:
        if expect_activation:
            if op not in _SINGLE_STAGE_ACTIVATIONS:
                return False
        elif op != 'Linear':
            return False
        expect_activation = not expect_activation
    return not expect_activation


def _gpu_native_training_plan(graph: NativeGraph) -> dict[str, Any]:
    nodes = list(graph.topological_order())
    ops = [node.op_type for node in nodes]
    if ops == ['Linear']:
        return {'kind': 'linear', 'linear_nodes': [nodes[0]]}
    if ops == ['Flatten', 'Linear']:
        return {'kind': 'linear', 'linear_nodes': [nodes[1]]}
    if len(ops) == 3 and ops[0] == 'Linear' and ops[1] in {'GELU', 'LeakyReLU', 'ReLU', 'SiLU', 'Sigmoid', 'Tanh'} and ops[2] == 'Linear':
        return {'kind': 'two_linear_activation', 'linear_nodes': [nodes[0], nodes[2]], 'activation_node': nodes[1]}
    if len(ops) == 4 and ops[0] == 'Flatten' and ops[1] == 'Linear' and ops[2] in {'GELU', 'LeakyReLU', 'ReLU', 'SiLU', 'Sigmoid', 'Tanh'} and ops[3] == 'Linear':
        return {'kind': 'two_linear_activation', 'linear_nodes': [nodes[1], nodes[3]], 'activation_node': nodes[2]}
    if ops == ['MaxPool2d', 'Flatten', 'Linear']:
        return {'kind': 'pool_linear', 'pool_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops == ['AvgPool2d', 'Flatten', 'Linear']:
        return {'kind': 'avgpool_linear', 'pool_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops == ['BatchNorm2d', 'Flatten', 'Linear']:
        return {'kind': 'batchnorm_linear', 'batchnorm_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops == ['Flatten', 'LayerNorm', 'Linear']:
        return {'kind': 'layernorm_linear', 'layernorm_node': nodes[1], 'linear_nodes': [nodes[2]]}
    if len(ops) == 4 and ops[0] == 'Flatten' and ops[1] == 'LayerNorm' and ops[2] in _SINGLE_STAGE_ACTIVATIONS and ops[3] == 'Linear':
        return {'kind': 'layernorm_activation_linear', 'layernorm_node': nodes[1], 'activation_node': nodes[2], 'linear_nodes': [nodes[3]]}
    if ops == ['LayerNorm2d', 'Flatten', 'Linear']:
        return {'kind': 'layernorm2d_linear', 'layernorm2d_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops == ['GroupNorm', 'Flatten', 'Linear']:
        return {'kind': 'groupnorm_linear', 'groupnorm_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops == ['DepthwiseConv2d', 'LayerNorm2d', 'Flatten', 'Linear']:
        conv_attrs = dict(getattr(nodes[0], 'attrs', {}) or {})

        def _pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        if bool(conv_attrs.get('bias', False)):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires bias=false.')
        if _pair(conv_attrs.get('stride', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires stride=1.')
        if _pair(conv_attrs.get('padding', 0), 0) != (0, 0):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires padding=0.')
        if _pair(conv_attrs.get('dilation', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires dilation=1.')
        return {
            'kind': 'depthwise_layernorm2d_linear',
            'conv_node': nodes[0],
            'layernorm2d_node': nodes[1],
            'linear_nodes': [nodes[3]],
        }
    if ops == ['DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear']:
        depthwise_attrs = dict(getattr(nodes[0], 'attrs', {}) or {})
        pointwise_attrs = dict(getattr(nodes[2], 'attrs', {}) or {})

        def _pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        if bool(depthwise_attrs.get('bias', False)):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires bias=false.')
        if _pair(depthwise_attrs.get('stride', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires stride=1.')
        if _pair(depthwise_attrs.get('padding', 0), 0) != (0, 0):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires padding=0.')
        if _pair(depthwise_attrs.get('dilation', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires dilation=1.')
        if bool(pointwise_attrs.get('bias', False)):
            raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires bias=false.')
        if _pair(pointwise_attrs.get('kernel_size', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires kernel_size=1.')
        if _pair(pointwise_attrs.get('stride', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires stride=1.')
        if _pair(pointwise_attrs.get('padding', 0), 0) != (0, 0):
            raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires padding=0.')
        return {
            'kind': 'depthwise_layernorm2d_pointwise_linear',
            'depthwise_node': nodes[0],
            'layernorm2d_node': nodes[1],
            'pointwise_node': nodes[2],
            'linear_nodes': [nodes[4]],
        }
    if (
        len(ops) == 7
        and ops[0:3] == ['DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d']
        and ops[3] in _SINGLE_STAGE_ACTIVATIONS
        and ops[4:] == ['PointwiseConv2d', 'Flatten', 'Linear']
    ):
        depthwise_attrs = dict(getattr(nodes[0], 'attrs', {}) or {})
        pointwise1_attrs = dict(getattr(nodes[2], 'attrs', {}) or {})
        pointwise2_attrs = dict(getattr(nodes[4], 'attrs', {}) or {})

        def _pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        if bool(depthwise_attrs.get('bias', False)):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires bias=false.')
        if _pair(depthwise_attrs.get('stride', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires stride=1.')
        if _pair(depthwise_attrs.get('padding', 0), 0) != (0, 0):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires padding=0.')
        if _pair(depthwise_attrs.get('dilation', 1), 1) != (1, 1):
            raise ValueError('cuda_native gpu_native DepthwiseConv2d train-native subset currently requires dilation=1.')
        for pointwise_attrs in (pointwise1_attrs, pointwise2_attrs):
            if bool(pointwise_attrs.get('bias', False)):
                raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires bias=false.')
            if _pair(pointwise_attrs.get('kernel_size', 1), 1) != (1, 1):
                raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires kernel_size=1.')
            if _pair(pointwise_attrs.get('stride', 1), 1) != (1, 1):
                raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires stride=1.')
            if _pair(pointwise_attrs.get('padding', 0), 0) != (0, 0):
                raise ValueError('cuda_native gpu_native PointwiseConv2d train-native subset currently requires padding=0.')
        return {
            'kind': 'depthwise_layernorm2d_pointwise_gelu_pointwise_linear',
            'depthwise_node': nodes[0],
            'layernorm2d_node': nodes[1],
            'pointwise1_node': nodes[2],
            'pointwise2_node': nodes[4],
            'activation_kind': str(nodes[3].op_type),
            'activation_alpha': (
                float(getattr(nodes[3], 'attrs', {}).get('negative_slope', 0.01))
                if str(nodes[3].op_type) == 'LeakyReLU'
                else 0.01
            ),
            'linear_nodes': [nodes[6]],
        }
    if ops in (['GlobalAvgPool2d', 'Flatten', 'Linear'], ['AdaptiveAvgPool2d', 'Flatten', 'Linear']):
        return {'kind': 'global_avgpool_linear', 'pool_node': nodes[0], 'linear_nodes': [nodes[2]]}
    if ops in (
        ['Conv2d', 'Flatten', 'Linear'],
        ['Conv2d', 'ReLU', 'Flatten', 'Linear'],
        ['Conv2d', 'LeakyReLU', 'Flatten', 'Linear'],
        ['Conv2d', 'GELU', 'Flatten', 'Linear'],
        ['Conv2d', 'SiLU', 'Flatten', 'Linear'],
        ['Conv2d', 'Sigmoid', 'Flatten', 'Linear'],
        ['Conv2d', 'Tanh', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'ReLU', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'LeakyReLU', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'GELU', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'SiLU', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'Sigmoid', 'Flatten', 'Linear'],
        ['PointwiseConv2d', 'Tanh', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'ReLU', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'LeakyReLU', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'GELU', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'SiLU', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'Sigmoid', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'Tanh', 'Flatten', 'Linear'],
        ['Conv2d', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'LeakyReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'GELU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'SiLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'Sigmoid', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'Tanh', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'LeakyReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'GELU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'SiLU', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'Sigmoid', 'MaxPool2d', 'Flatten', 'Linear'],
        ['DepthwiseConv2d', 'Tanh', 'MaxPool2d', 'Flatten', 'Linear'],
        ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
    ):
        conv_attrs = dict(getattr(nodes[0], 'attrs', {}) or {})
        conv_op = str(nodes[0].op_type)

        def _pair(value: Any, default: int) -> tuple[int, int]:
            if value is None:
                return (default, default)
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return (int(value[0]), int(value[0]))
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        if bool(conv_attrs.get('bias', False)):
            raise ValueError(f'cuda_native gpu_native {conv_op} train-native subset currently requires bias=false.')
        if conv_op != 'DepthwiseConv2d' and int(conv_attrs.get('groups', 1)) != 1:
            raise ValueError(f'cuda_native gpu_native {conv_op} train-native subset currently requires groups=1.')
        if _pair(conv_attrs.get('stride', 1), 1) != (1, 1):
            raise ValueError(f'cuda_native gpu_native {conv_op} train-native subset currently requires stride=1.')
        if _pair(conv_attrs.get('padding', 0), 0) != (0, 0):
            raise ValueError(f'cuda_native gpu_native {conv_op} train-native subset currently requires padding=0.')
        if _pair(conv_attrs.get('dilation', 1), 1) != (1, 1):
            raise ValueError(f'cuda_native gpu_native {conv_op} train-native subset currently requires dilation=1.')
        if (
            len(ops) == 7
            and ops[0] == 'Conv2d'
            and ops[1] in _SINGLE_STAGE_ACTIVATIONS
            and ops[2] == 'Conv2d'
            and ops[3] == ops[1]
            and ops[4:] == ['MaxPool2d', 'Flatten', 'Linear']
        ):
            conv2_attrs = dict(getattr(nodes[2], 'attrs', {}) or {})
            if bool(conv2_attrs.get('bias', False)):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires bias=false.')
            if int(conv2_attrs.get('groups', 1)) != 1:
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires groups=1.')
            if _pair(conv2_attrs.get('stride', 1), 1) != (1, 1):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires stride=1.')
            if _pair(conv2_attrs.get('padding', 0), 0) != (0, 0):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires padding=0.')
            if _pair(conv2_attrs.get('dilation', 1), 1) != (1, 1):
                raise ValueError('cuda_native gpu_native Conv2d train-native subset currently requires dilation=1.')
            return {
                'kind': 'two_conv_relu_pool_linear',
                'conv_nodes': [nodes[0], nodes[2]],
                'linear_nodes': [nodes[-1]],
                'activation_kind': str(nodes[1].op_type),
                'activation_alpha': (
                    float(getattr(nodes[1], 'attrs', {}).get('negative_slope', 0.01))
                    if str(nodes[1].op_type) == 'LeakyReLU'
                    else 0.01
                ),
            }
        activation_node = next((node for node in nodes if node.op_type in _SINGLE_STAGE_ACTIVATIONS), None)
        activation_kind = None if activation_node is None else str(activation_node.op_type)
        has_pool = 'MaxPool2d' in ops
        linear_node = nodes[-1]
        return {
            'kind': 'conv_linear',
            'conv_node': nodes[0],
            'linear_nodes': [linear_node],
            'apply_relu_activation': activation_kind == 'ReLU',
            'activation_kind': activation_kind,
            'activation_alpha': (
                float(getattr(activation_node, 'attrs', {}).get('negative_slope', 0.01))
                if activation_kind == 'LeakyReLU' and activation_node is not None
                else 0.01
            ),
            'apply_maxpool': has_pool,
            'conv_kind': 'depthwise' if conv_op == 'DepthwiseConv2d' else 'conv2d',
        }
    if _is_generic_mlp_ops(ops):
        return {
            'kind': 'generic_mlp',
            'linear_nodes': [node for node in nodes if node.op_type == 'Linear'],
            'activation_nodes': [node for node in nodes if node.op_type in _SINGLE_STAGE_ACTIVATIONS],
        }
    raise ValueError(
        'cuda_native gpu_native train-native currently supports only '
        'ops=[Linear], ops=[Flatten, Linear], ops=[Linear, ReLU, Linear], '
        'ops=[Flatten, Linear, ReLU, Linear], '
        'ops=[Linear, LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Linear], '
        'ops=[Flatten, Linear, LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Linear], '
        'ops=[MaxPool2d, Flatten, Linear], '
        'ops=[GlobalAvgPool2d, Flatten, Linear], ops=[AdaptiveAvgPool2d, Flatten, Linear], '
        'ops=[DepthwiseConv2d, LayerNorm2d, Flatten, Linear], '
        'ops=[Conv2d, Flatten, Linear], '
        'ops=[Conv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Flatten, Linear], '
        'ops=[Conv2d, MaxPool2d, Flatten, Linear], or '
        'ops=[Conv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, MaxPool2d, Flatten, Linear], or '
        'ops=[Conv2d, ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, Conv2d, same activation, MaxPool2d, Flatten, Linear], '
        'or generic sequential MLP ops=[Flatten?, Linear, activation, Linear, ..., Linear], '
        f'got {ops}.'
    )


def _validate_gpu_native_training_context(ctx: NativeTrainingContext) -> None:
    plan = _gpu_native_training_plan(ctx.graph)
    if ctx.loss_type != 'cross_entropy' and plan['kind'] != 'linear':
        raise ValueError('cuda_native gpu_native train-native currently supports MSELoss/BCEWithLogitsLoss only for the Linear subset.')
    if plan['kind'] == 'linear':
        if ctx.optimizer_type not in {'sgd', 'adam', 'adamw', 'rmsprop'}:
            raise ValueError('cuda_native gpu_native Linear train-native currently supports optimizer.type in {SGD, Adam, AdamW, RMSprop}.')
        if ctx.optimizer_type == 'adam' and ctx.weight_decay != 0.0:
            raise ValueError('cuda_native gpu_native Linear train-native currently requires Adam weight_decay=0.0; use AdamW for decoupled weight decay.')
    else:
        if ctx.optimizer_type != 'sgd':
            raise ValueError('cuda_native gpu_native non-Linear train-native currently supports only optimizer.type=SGD.')
    if ctx.amp:
        raise ValueError('cuda_native gpu_native train-native currently requires train.amp=false.')


def _merge_gpu_native_step_runtime(ctx: NativeTrainingContext, step_summary: dict[str, Any]) -> None:
    for attr in (
        'host_to_device_transfer_events',
        'host_to_device_transfer_bytes',
        'device_to_host_transfer_events',
        'device_to_host_transfer_bytes',
        'allocation_events',
        'allocated_bytes',
        'synchronization_events',
        'device_pointer_allocation_events',
        'device_pointer_free_events',
        'device_pointer_bytes',
        'device_pointer_live_bytes',
        'device_sync_to_host_events',
        'device_sync_to_device_events',
        'persistent_device_cache_hits',
        'persistent_device_cache_misses',
        'persistent_device_cache_invalidations',
    ):
        setattr(ctx.device_runtime, attr, int(getattr(ctx.device_runtime, attr)) + int(step_summary.get(attr, 0)))
    ctx.device_runtime.persistent_device_cache_entries = int(step_summary.get('persistent_device_cache_entries', getattr(ctx.device_runtime, 'persistent_device_cache_entries', 0)))
    ctx.device_runtime.persistent_device_cache_bytes = int(step_summary.get('persistent_device_cache_bytes', getattr(ctx.device_runtime, 'persistent_device_cache_bytes', 0)))
    for kind, count in dict(step_summary.get('execution_kinds', {})).items():
        for _ in range(int(count)):
            ctx.device_runtime.record_execution(str(kind), node_count=0)
