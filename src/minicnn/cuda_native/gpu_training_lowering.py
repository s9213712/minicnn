from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minicnn.cuda_native.gpu_dispatch import GpuDispatchStep, build_gpu_dispatch_plan
from minicnn.cuda_native.graph import NativeGraph


@dataclass(frozen=True)
class GpuTrainingLoweringStep:
    phase: str
    op_name: str
    lowering_kind: str
    launch_family: str
    node_name: str | None = None
    param_keys: tuple[str, ...] = tuple()
    required_symbols: tuple[str, ...] = tuple()
    supported: bool = True

    def __post_init__(self) -> None:
        if not self.required_symbols:
            object.__setattr__(
                self,
                'required_symbols',
                _required_symbols_for_lowering(self.lowering_kind),
            )

    def summary(self) -> dict[str, Any]:
        return {
            'phase': self.phase,
            'op_name': self.op_name,
            'lowering_kind': self.lowering_kind,
            'launch_family': self.launch_family,
            'node_name': self.node_name,
            'param_keys': list(self.param_keys),
            'required_symbols': list(self.required_symbols),
            'supported': self.supported,
        }


@dataclass(frozen=True)
class GpuTrainingLoweringPlan:
    execution_mode: str
    ready: bool
    subset_name: str | None
    helper: str | None
    forward_steps: tuple[GpuTrainingLoweringStep, ...]
    loss_step: GpuTrainingLoweringStep | None
    backward_steps: tuple[GpuTrainingLoweringStep, ...]
    optimizer_steps: tuple[GpuTrainingLoweringStep, ...]
    unsupported_reasons: tuple[str, ...]

    def _steps_by_phase(self) -> dict[str, tuple[GpuTrainingLoweringStep, ...]]:
        return {
            'forward': self.forward_steps,
            'loss': tuple() if self.loss_step is None else (self.loss_step,),
            'backward': self.backward_steps,
            'optimizer': self.optimizer_steps,
        }

    def required_symbols_by_phase(self) -> dict[str, list[str]]:
        return {
            phase: sorted({
                symbol
                for step in steps
                for symbol in step.required_symbols
            })
            for phase, steps in self._steps_by_phase().items()
        }

    def required_symbols(self) -> list[str]:
        symbols: set[str] = set()
        for phase_symbols in self.required_symbols_by_phase().values():
            symbols.update(phase_symbols)
        return sorted(symbols)

    def summary(self) -> dict[str, Any]:
        return {
            'execution_mode': self.execution_mode,
            'ready': self.ready,
            'subset_name': self.subset_name,
            'helper': self.helper,
            'required_symbols': self.required_symbols(),
            'required_symbols_by_phase': self.required_symbols_by_phase(),
            'forward_steps': [step.summary() for step in self.forward_steps],
            'loss_step': None if self.loss_step is None else self.loss_step.summary(),
            'backward_steps': [step.summary() for step in self.backward_steps],
            'optimizer_steps': [step.summary() for step in self.optimizer_steps],
            'unsupported_reasons': list(self.unsupported_reasons),
        }


def _required_symbols_for_lowering(lowering_kind: str) -> tuple[str, ...]:
    return {
        'shape_flatten_shim': tuple(),
        'shape_identity_alias_shim': tuple(),
        'identity_alias_shim': tuple(),
        'regularization_dropout_p0_alias_shim': tuple(),
        'regularization_droppath_p0_alias_shim': tuple(),
        'linear_affine_shim': ('dense_forward',),
        'activation_relu_shim': ('apply_relu',),
        'activation_leaky_relu_shim': ('leaky_relu_forward',),
        'activation_sigmoid_shim': ('sigmoid_forward',),
        'activation_tanh_shim': ('tanh_forward',),
        'activation_silu_shim': ('silu_forward',),
        'activation_gelu_shim': ('gelu_forward',),
        'elementwise_relu_shim': ('apply_relu',),
        'elementwise_leaky_relu_shim': ('leaky_relu_forward',),
        'elementwise_sigmoid_shim': ('sigmoid_forward',),
        'elementwise_tanh_shim': ('tanh_forward',),
        'elementwise_silu_shim': ('silu_forward',),
        'elementwise_gelu_shim': ('gelu_forward',),
        'merge_add_shim': ('add_forward',),
        'merge_concat_shim': ('concat_forward',),
        'elementwise_add_shim': ('add_forward',),
        'concat_shim': ('concat_forward',),
        'pool_maxpool2d_shim': ('apply_maxpool',),
        'pool_avgpool2d_shim': ('avgpool2d_forward',),
        'pool_global_avgpool2d_shim': ('global_avgpool2d_forward',),
        'pool2d_shim': ('apply_maxpool',),
        'avgpool2d_shim': ('avgpool2d_forward',),
        'global_avgpool2d_shim': ('global_avgpool2d_forward',),
        'normalization_batchnorm2d_shim': ('bn_eval_forward',),
        'normalization_layernorm2d_shim': ('layernorm2d_forward',),
        'normalization_groupnorm_shim': ('groupnorm_forward',),
        'batchnorm2d_shim': ('bn_eval_forward',),
        'layernorm2d_shim': ('layernorm2d_forward',),
        'groupnorm_shim': ('groupnorm_forward',),
        'conv2d_reference_shim': ('im2col_forward', 'gemm_forward', 'cnhw_to_nchw'),
        'conv2d_shim': ('im2col_forward', 'gemm_forward', 'cnhw_to_nchw'),
        'depthwise_conv2d_shim': ('depthwise_conv2d_forward',),
        'softmax_xent_grad_loss_acc': ('softmax_xent_grad_loss_acc',),
        'softmax_xent_smooth_grad_loss_acc': ('softmax_xent_smooth_grad_loss_acc',),
        'mse_fwd_grad_loss_acc': ('mse_fwd_grad_loss_acc',),
        'bce_fwd_grad_loss_acc': ('bce_fwd_grad_loss_acc',),
        'dense_backward_full': ('dense_backward_full',),
        'apply_relu_backward': ('apply_relu_backward',),
        'gelu_backward': ('gelu_backward',),
        'silu_backward': ('silu_backward',),
        'sigmoid_backward': ('sigmoid_backward',),
        'tanh_backward': ('tanh_backward',),
        'maxpool_backward_nchw': ('maxpool_backward_nchw',),
        'avgpool2d_backward': ('avgpool2d_backward',),
        'global_avgpool2d_backward': ('global_avgpool2d_backward',),
        'bn_backward': ('bn_backward',),
        'layernorm2d_backward': ('layernorm2d_backward',),
        'groupnorm_backward': ('groupnorm_backward',),
        'conv_backward': ('conv_backward',),
        'depthwise_conv2d_backward': ('depthwise_conv2d_backward',),
        'grad_l2_sumsq_scale': ('grad_l2_sumsq', 'scale_inplace'),
        'apply_sgd_update': ('apply_sgd_update',),
        'apply_momentum_update': ('apply_momentum_update',),
        'sgd_update_fused': ('sgd_update_fused',),
        'adam_update_fused': ('adam_update_fused',),
        'rmsprop_update_fused': ('rmsprop_update_fused',),
    }.get(str(lowering_kind), tuple())


_TRAINING_SUBSETS: dict[tuple[str, ...], tuple[str, str]] = {
    ('Linear',): ('linear', 'native_gpu_linear_training_step'),
    ('Flatten', 'Linear'): ('flatten_linear', 'native_gpu_linear_training_step'),
    ('Linear', 'ReLU', 'Linear'): ('linear_relu_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Flatten', 'Linear', 'ReLU', 'Linear'): ('flatten_linear_relu_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Linear', 'GELU', 'Linear'): ('linear_gelu_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Flatten', 'Linear', 'GELU', 'Linear'): ('flatten_linear_gelu_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Linear', 'SiLU', 'Linear'): ('linear_silu_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Flatten', 'Linear', 'SiLU', 'Linear'): ('flatten_linear_silu_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Linear', 'Sigmoid', 'Linear'): ('linear_sigmoid_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Flatten', 'Linear', 'Sigmoid', 'Linear'): ('flatten_linear_sigmoid_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Linear', 'Tanh', 'Linear'): ('linear_tanh_linear', 'native_gpu_two_linear_relu_training_step'),
    ('Flatten', 'Linear', 'Tanh', 'Linear'): ('flatten_linear_tanh_linear', 'native_gpu_two_linear_relu_training_step'),
    ('MaxPool2d', 'Flatten', 'Linear'): ('maxpool_linear', 'native_gpu_pool_linear_training_step'),
    ('AvgPool2d', 'Flatten', 'Linear'): ('avgpool_linear', 'native_gpu_avgpool_linear_training_step'),
    ('BatchNorm2d', 'Flatten', 'Linear'): ('batchnorm_linear', 'native_gpu_batchnorm_linear_training_step'),
    ('LayerNorm2d', 'Flatten', 'Linear'): ('layernorm2d_linear', 'native_gpu_layernorm2d_linear_training_step'),
    ('GroupNorm', 'Flatten', 'Linear'): ('groupnorm_linear', 'native_gpu_groupnorm_linear_training_step'),
    ('DepthwiseConv2d', 'LayerNorm2d', 'Flatten', 'Linear'): (
        'depthwise_layernorm2d_linear',
        'native_gpu_depthwise_layernorm2d_linear_training_step',
    ),
    ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear'): (
        'depthwise_layernorm2d_pointwise_linear',
        'native_gpu_depthwise_layernorm2d_pointwise_linear_training_step',
    ),
    ('GlobalAvgPool2d', 'Flatten', 'Linear'): ('global_avgpool_linear', 'native_gpu_global_avgpool_linear_training_step'),
    ('AdaptiveAvgPool2d', 'Flatten', 'Linear'): ('adaptive_avgpool_linear', 'native_gpu_global_avgpool_linear_training_step'),
    ('Conv2d', 'Flatten', 'Linear'): ('conv_linear', 'native_gpu_conv_linear_training_step'),
    ('Conv2d', 'ReLU', 'Flatten', 'Linear'): ('conv_relu_linear', 'native_gpu_conv_linear_training_step'),
    ('PointwiseConv2d', 'Flatten', 'Linear'): ('pointwise_conv_linear', 'native_gpu_conv_linear_training_step'),
    ('PointwiseConv2d', 'ReLU', 'Flatten', 'Linear'): ('pointwise_conv_relu_linear', 'native_gpu_conv_linear_training_step'),
    ('DepthwiseConv2d', 'Flatten', 'Linear'): ('depthwise_conv_linear', 'native_gpu_conv_linear_training_step'),
    ('DepthwiseConv2d', 'ReLU', 'Flatten', 'Linear'): ('depthwise_conv_relu_linear', 'native_gpu_conv_linear_training_step'),
    ('DepthwiseConv2d', 'MaxPool2d', 'Flatten', 'Linear'): ('depthwise_conv_pool_linear', 'native_gpu_conv_linear_training_step'),
    ('DepthwiseConv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'): (
        'depthwise_conv_relu_pool_linear',
        'native_gpu_conv_linear_training_step',
    ),
    ('Conv2d', 'MaxPool2d', 'Flatten', 'Linear'): ('conv_pool_linear', 'native_gpu_conv_linear_training_step'),
    ('Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'): (
        'conv_relu_pool_linear',
        'native_gpu_conv_linear_training_step',
    ),
    ('Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'): (
        'two_conv_relu_pool_linear',
        'native_gpu_two_conv_relu_pool_linear_training_step',
    ),
}


def _forward_training_step(step: GpuDispatchStep) -> GpuTrainingLoweringStep:
    return GpuTrainingLoweringStep(
        phase='forward',
        op_name=step.op_name,
        lowering_kind=step.lowering_kind,
        launch_family=step.launch_family,
        node_name=step.node_name,
        param_keys=step.param_keys,
        supported=step.supported,
    )


def _linear_nodes(graph: NativeGraph) -> list[Any]:
    return [node for node in graph.topological_order() if node.op_type == 'Linear']


def _conv_nodes(graph: NativeGraph) -> list[Any]:
    return [node for node in graph.topological_order() if node.op_type in {'Conv2d', 'DepthwiseConv2d', 'PointwiseConv2d'}]


def _loss_step(
    loss_type: str,
    subset_name: str | None,
    *,
    label_smoothing: float = 0.0,
) -> tuple[GpuTrainingLoweringStep | None, list[str]]:
    reasons: list[str] = []
    if loss_type == 'CrossEntropyLoss':
        if float(label_smoothing) > 0.0:
            return GpuTrainingLoweringStep(
                phase='loss',
                op_name='CrossEntropyLoss',
                lowering_kind='softmax_xent_smooth_grad_loss_acc',
                launch_family='loss_softmax_xent_smooth',
            ), reasons
        return GpuTrainingLoweringStep(
            phase='loss',
            op_name='CrossEntropyLoss',
            lowering_kind='softmax_xent_grad_loss_acc',
            launch_family='loss_softmax_xent',
        ), reasons
    if subset_name in {'linear', 'flatten_linear'} and loss_type == 'MSELoss':
        return GpuTrainingLoweringStep(
            phase='loss',
            op_name='MSELoss',
            lowering_kind='mse_fwd_grad_loss_acc',
            launch_family='loss_mse',
        ), reasons
    if subset_name in {'linear', 'flatten_linear'} and loss_type == 'BCEWithLogitsLoss':
        return GpuTrainingLoweringStep(
            phase='loss',
            op_name='BCEWithLogitsLoss',
            lowering_kind='bce_fwd_grad_loss_acc',
            launch_family='loss_bce_with_logits',
        ), reasons
    reasons.append(f'unsupported gpu_native training loss: {loss_type}')
    return None, reasons


def _backward_steps(graph: NativeGraph, subset_name: str | None) -> tuple[GpuTrainingLoweringStep, ...]:
    linear_nodes = _linear_nodes(graph)
    conv_nodes = _conv_nodes(graph)
    steps: list[GpuTrainingLoweringStep] = []

    for node in reversed(linear_nodes):
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='Linear',
                lowering_kind='dense_backward_full',
                launch_family='linear_backward',
                node_name=str(node.name),
                param_keys=(f'_w_{node.name}', f'_b_{node.name}'),
            )
        )
    activation_nodes = [
        node for node in graph.topological_order()
        if node.op_type in {'GELU', 'ReLU', 'SiLU', 'Sigmoid', 'Tanh'}
    ]
    if subset_name in {
        'linear_relu_linear',
        'flatten_linear_relu_linear',
        'linear_gelu_linear',
        'flatten_linear_gelu_linear',
        'linear_silu_linear',
        'flatten_linear_silu_linear',
        'linear_sigmoid_linear',
        'flatten_linear_sigmoid_linear',
        'linear_tanh_linear',
        'flatten_linear_tanh_linear',
    } and activation_nodes:
        activation_op = str(activation_nodes[0].op_type)
        activation_lowering = 'apply_relu_backward' if activation_op == 'ReLU' else f'{activation_op.lower()}_backward'
        steps.insert(
            1,
            GpuTrainingLoweringStep(
                phase='backward',
                op_name=activation_op,
                lowering_kind=activation_lowering,
                launch_family='activation_backward',
            ),
        )
    if subset_name == 'maxpool_linear':
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='MaxPool2d',
                lowering_kind='maxpool_backward_nchw',
                launch_family='pool_backward',
            )
        )
    if subset_name == 'avgpool_linear':
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='AvgPool2d',
                lowering_kind='avgpool2d_backward',
                launch_family='pool_backward',
            )
        )
    if subset_name in {'global_avgpool_linear', 'adaptive_avgpool_linear'}:
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='GlobalAvgPool2d',
                lowering_kind='global_avgpool2d_backward',
                launch_family='pool_backward',
            )
        )
    if subset_name == 'depthwise_layernorm2d_pointwise_linear':
        pointwise_node = next(node for node in conv_nodes if str(node.op_type) == 'PointwiseConv2d')
        depthwise_node = next(node for node in conv_nodes if str(node.op_type) == 'DepthwiseConv2d')
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='PointwiseConv2d',
                lowering_kind='conv_backward',
                launch_family='conv2d_backward',
                node_name=str(pointwise_node.name),
                param_keys=(f'_w_{pointwise_node.name}',),
            )
        )
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='LayerNorm2d',
                lowering_kind='layernorm2d_backward',
                launch_family='normalization_backward',
            )
        )
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='DepthwiseConv2d',
                lowering_kind='depthwise_conv2d_backward',
                launch_family='depthwise_conv2d_backward',
                node_name=str(depthwise_node.name),
                param_keys=(f'_w_{depthwise_node.name}',),
            )
        )
        return tuple(steps)
    if subset_name == 'batchnorm_linear':
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='BatchNorm2d',
                lowering_kind='bn_backward',
                launch_family='normalization_backward',
            )
        )
    if subset_name in {
        'layernorm2d_linear',
        'depthwise_layernorm2d_linear',
        'depthwise_layernorm2d_pointwise_linear',
    }:
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='LayerNorm2d',
                lowering_kind='layernorm2d_backward',
                launch_family='normalization_backward',
            )
        )
    if subset_name == 'groupnorm_linear':
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='GroupNorm',
                lowering_kind='groupnorm_backward',
                launch_family='normalization_backward',
            )
        )
    if subset_name in {'conv_relu_linear', 'pointwise_conv_relu_linear', 'depthwise_conv_relu_linear', 'conv_relu_pool_linear', 'depthwise_conv_relu_pool_linear', 'two_conv_relu_pool_linear'}:
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='ReLU',
                lowering_kind='apply_relu_backward',
                launch_family='activation_backward',
            )
        )
    if subset_name in {'conv_pool_linear', 'conv_relu_pool_linear', 'depthwise_conv_pool_linear', 'depthwise_conv_relu_pool_linear', 'two_conv_relu_pool_linear'}:
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name='MaxPool2d',
                lowering_kind='maxpool_backward_nchw',
                launch_family='pool_backward',
            )
        )
    for node in reversed(conv_nodes):
        is_depthwise = str(node.op_type) == 'DepthwiseConv2d'
        steps.append(
            GpuTrainingLoweringStep(
                phase='backward',
                op_name=str(node.op_type),
                lowering_kind='depthwise_conv2d_backward' if is_depthwise else 'conv_backward',
                launch_family='depthwise_conv2d_backward' if is_depthwise else 'conv2d_backward',
                node_name=str(node.name),
                param_keys=(f'_w_{node.name}',),
            )
        )
    return tuple(steps)


def _optimizer_step(
    graph: NativeGraph,
    subset_name: str | None,
    optim_cfg: dict[str, Any],
) -> tuple[tuple[GpuTrainingLoweringStep, ...], list[str]]:
    reasons: list[str] = []
    optimizer_type = str(optim_cfg.get('type', 'SGD'))
    optimizer_key = optimizer_type.lower()
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    momentum = float(optim_cfg.get('momentum', 0.0))
    param_keys: list[str] = []
    for node in graph.topological_order():
        if node.op_type in {'BatchNorm2d', 'Conv2d', 'DepthwiseConv2d', 'GroupNorm', 'LayerNorm2d', 'Linear', 'PointwiseConv2d'}:
            param_keys.append(f'_w_{node.name}')
            if node.op_type in {'BatchNorm2d', 'GroupNorm', 'LayerNorm2d', 'Linear'} or bool(node.attrs.get('bias', True)):
                param_keys.append(f'_b_{node.name}')

    if subset_name not in {'linear', 'flatten_linear'} and optimizer_key != 'sgd':
        reasons.append(f'unsupported gpu_native optimizer for non-Linear subset: {optimizer_type}')
        return tuple(), reasons
    if optimizer_key == 'adam' and weight_decay != 0.0:
        reasons.append('gpu_native Linear Adam requires weight_decay=0.0; use AdamW for decoupled weight decay')
        return tuple(), reasons
    if optimizer_key == 'sgd':
        if weight_decay != 0.0 or (subset_name not in {'linear', 'flatten_linear'} and momentum != 0.0):
            lowering_kind = 'sgd_update_fused'
            launch_family = 'optimizer_sgd_fused'
        elif momentum != 0.0:
            lowering_kind = 'apply_momentum_update'
            launch_family = 'optimizer_momentum'
        else:
            lowering_kind = 'apply_sgd_update'
            launch_family = 'optimizer_sgd'
    elif optimizer_key in {'adam', 'adamw'} and subset_name in {'linear', 'flatten_linear'}:
        lowering_kind = 'adam_update_fused'
        launch_family = 'optimizer_adam'
    elif optimizer_key == 'rmsprop' and subset_name in {'linear', 'flatten_linear'}:
        lowering_kind = 'rmsprop_update_fused'
        launch_family = 'optimizer_rmsprop'
    else:
        reasons.append(f'unsupported gpu_native optimizer: {optimizer_type}')
        return tuple(), reasons

    steps: list[GpuTrainingLoweringStep] = []
    if float(optim_cfg.get('grad_clip_global', 0.0)) != 0.0:
        steps.append(
            GpuTrainingLoweringStep(
                phase='optimizer',
                op_name='GlobalGradClip',
                lowering_kind='grad_l2_sumsq_scale',
                launch_family='optimizer_global_norm_clip',
                param_keys=tuple(param_keys),
            )
        )
    steps.append(
        GpuTrainingLoweringStep(
            phase='optimizer',
            op_name=optimizer_type,
            lowering_kind=lowering_kind,
            launch_family=launch_family,
            param_keys=tuple(param_keys),
        )
    )
    return tuple(steps), reasons


def build_gpu_training_lowering_plan(
    graph: NativeGraph,
    *,
    loss_cfg: dict[str, Any] | None = None,
    optim_cfg: dict[str, Any] | None = None,
    train_cfg: dict[str, Any] | None = None,
) -> GpuTrainingLoweringPlan:
    loss_cfg = dict(loss_cfg or {})
    optim_cfg = dict(optim_cfg or {})
    train_cfg = dict(train_cfg or {})
    dispatch_plan = build_gpu_dispatch_plan(graph)
    ops = tuple(node.op_type for node in graph.topological_order())
    subset = _TRAINING_SUBSETS.get(ops)
    subset_name = None if subset is None else subset[0]
    helper = None if subset is None else subset[1]
    unsupported_reasons: list[str] = []
    if subset is None:
        unsupported_reasons.append(f'unsupported gpu_native training subset: {list(ops)}')
    if not dispatch_plan.ready:
        unsupported_reasons.append(f'unsupported gpu dispatch ops: {list(dispatch_plan.unsupported_ops)}')

    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    loss_step, loss_reasons = _loss_step(
        loss_type,
        subset_name,
        label_smoothing=float(loss_cfg.get('label_smoothing', 0.0)),
    )
    unsupported_reasons.extend(loss_reasons)

    optimizer_steps, optimizer_reasons = _optimizer_step(graph, subset_name, optim_cfg)
    unsupported_reasons.extend(optimizer_reasons)

    if bool(train_cfg.get('amp', False)):
        unsupported_reasons.append('gpu_native training lowering currently requires train.amp=false')
    forward_steps = tuple(_forward_training_step(step) for step in dispatch_plan.steps)
    backward_steps = _backward_steps(graph, subset_name) if subset_name is not None else tuple()
    ready = (
        dispatch_plan.ready
        and subset_name is not None
        and loss_step is not None
        and bool(optimizer_steps)
        and not unsupported_reasons
    )
    return GpuTrainingLoweringPlan(
        execution_mode='gpu_native',
        ready=ready,
        subset_name=subset_name,
        helper=helper,
        forward_steps=forward_steps,
        loss_step=loss_step,
        backward_steps=backward_steps,
        optimizer_steps=optimizer_steps,
        unsupported_reasons=tuple(unsupported_reasons),
    )
