from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs
from minicnn.cuda_native.gpu_lowering import list_gpu_lowering_specs
from minicnn.cuda_native.graph import NativeGraph


@dataclass(frozen=True)
class GpuLaunchDescriptor:
    launch_family: str
    input_bindings: tuple[str, ...]
    output_bindings: tuple[str, ...]
    param_bindings: tuple[str, ...]
    attr_bindings: dict[str, Any]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    tensor_dtype: str
    param_layouts: dict[str, str]
    normalized_tensor_args: tuple[dict[str, Any], ...]
    normalized_scalar_args: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GpuLaunchPacket:
    node_name: str
    op_name: str
    launch_family: str
    lowering_kind: str
    preferred_layout: str
    tensor_args: tuple[dict[str, Any], ...]
    scalar_args: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {
            'node_name': self.node_name,
            'op_name': self.op_name,
            'launch_family': self.launch_family,
            'lowering_kind': self.lowering_kind,
            'preferred_layout': self.preferred_layout,
            'tensor_args': [dict(arg) for arg in self.tensor_args],
            'scalar_args': [dict(arg) for arg in self.scalar_args],
        }


@dataclass(frozen=True)
class GpuDispatchStep:
    node_name: str
    op_name: str
    category: str
    launch_family: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    param_keys: tuple[str, ...]
    input_arity: int
    output_arity: int
    preferred_layout: str
    lowering_kind: str
    launch_descriptor: GpuLaunchDescriptor
    forward_status: str
    backward_status: str
    supported: bool = True


@dataclass(frozen=True)
class GpuDispatchPlan:
    execution_mode: str
    ready: bool
    steps: tuple[GpuDispatchStep, ...]
    unsupported_ops: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            'execution_mode': self.execution_mode,
            'ready': self.ready,
            'num_steps': len(self.steps),
            'unsupported_ops': list(self.unsupported_ops),
            'steps': [
                {
                    'node_name': step.node_name,
                    'op_name': step.op_name,
                    'category': step.category,
                    'launch_family': step.launch_family,
                    'input_names': list(step.input_names),
                    'output_names': list(step.output_names),
                    'param_keys': list(step.param_keys),
                    'input_arity': step.input_arity,
                    'output_arity': step.output_arity,
                    'preferred_layout': step.preferred_layout,
                    'lowering_kind': step.lowering_kind,
                    'launch_descriptor': {
                        'launch_family': step.launch_descriptor.launch_family,
                        'input_bindings': list(step.launch_descriptor.input_bindings),
                        'output_bindings': list(step.launch_descriptor.output_bindings),
                        'param_bindings': list(step.launch_descriptor.param_bindings),
                        'attr_bindings': dict(step.launch_descriptor.attr_bindings),
                        'input_shapes': [list(shape) for shape in step.launch_descriptor.input_shapes],
                        'output_shapes': [list(shape) for shape in step.launch_descriptor.output_shapes],
                        'tensor_dtype': step.launch_descriptor.tensor_dtype,
                        'param_layouts': dict(step.launch_descriptor.param_layouts),
                        'normalized_tensor_args': [dict(arg) for arg in step.launch_descriptor.normalized_tensor_args],
                        'normalized_scalar_args': [dict(arg) for arg in step.launch_descriptor.normalized_scalar_args],
                    },
                    'forward_status': step.forward_status,
                    'backward_status': step.backward_status,
                    'supported': step.supported,
                }
                for step in self.steps
            ],
        }


def _node_param_keys(node) -> tuple[str, ...]:
    keys: list[str] = []
    if node.op_type in {'Conv2d', 'DepthwiseConv2d', 'PointwiseConv2d', 'Linear'}:
        keys.append(f'_w_{node.name}')
        if bool(node.attrs.get('bias', True)):
            keys.append(f'_b_{node.name}')
    elif node.op_type == 'BatchNorm2d':
        keys.extend((
            f'_w_{node.name}',
            f'_b_{node.name}',
            f'_running_mean_{node.name}',
            f'_running_var_{node.name}',
        ))
    elif node.op_type in {'GroupNorm', 'LayerNorm2d'}:
        keys.extend((f'_w_{node.name}', f'_b_{node.name}'))
    return tuple(keys)


def _node_attr_bindings(node) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    if node.op_type in {'Conv2d', 'DepthwiseConv2d', 'PointwiseConv2d'}:
        bindings['stride'] = node.attrs.get('stride', 1)
        bindings['padding'] = node.attrs.get('padding', 0)
        bindings['groups'] = int(node.attrs.get('groups', 1))
    elif node.op_type in {'MaxPool2d', 'AvgPool2d'}:
        bindings['kernel_size'] = node.attrs.get('kernel_size', 2)
        bindings['stride'] = node.attrs.get('stride', node.attrs.get('kernel_size', 2))
        bindings['padding'] = node.attrs.get('padding', 0)
    elif node.op_type == 'LeakyReLU':
        bindings['negative_slope'] = float(node.attrs.get('negative_slope', 0.01))
    elif node.op_type == 'Concat':
        bindings['axis'] = int(node.attrs.get('axis', 1))
    elif node.op_type == 'BatchNorm2d':
        bindings['eps'] = float(node.attrs.get('eps', 1e-5))
        bindings['momentum'] = float(node.attrs.get('momentum', 0.1))
    elif node.op_type == 'LayerNorm2d':
        bindings['eps'] = float(node.attrs.get('eps', 1e-6))
    elif node.op_type == 'GroupNorm':
        bindings['num_groups'] = int(node.attrs.get('num_groups', 1))
        bindings['eps'] = float(node.attrs.get('eps', 1e-5))
    elif node.op_type == 'AdaptiveAvgPool2d':
        bindings['output_size'] = node.attrs.get('output_size', 1)
    return bindings


def _node_param_layouts(node, param_keys: tuple[str, ...]) -> dict[str, str]:
    layouts: dict[str, str] = {}
    if node.op_type in {'Conv2d', 'DepthwiseConv2d', 'PointwiseConv2d'}:
        for key in param_keys:
            layouts[key] = 'OIHW' if key.startswith('_w_') else 'O'
    elif node.op_type == 'Linear':
        for key in param_keys:
            layouts[key] = 'OI' if key.startswith('_w_') else 'O'
    elif node.op_type == 'BatchNorm2d':
        for key in param_keys:
            layouts[key] = 'C'
    elif node.op_type in {'GroupNorm', 'LayerNorm2d'}:
        for key in param_keys:
            layouts[key] = 'C'
    else:
        for key in param_keys:
            layouts[key] = 'match_op_default'
    return layouts


def _normalized_tensor_args(
    node,
    input_bindings: tuple[str, ...],
    output_bindings: tuple[str, ...],
    param_bindings: tuple[str, ...],
    param_layouts: dict[str, str],
) -> tuple[dict[str, Any], ...]:
    args: list[dict[str, Any]] = []
    for idx, (binding, spec) in enumerate(zip(input_bindings, node.input_specs)):
        args.append({
            'kind': 'input',
            'index': idx,
            'binding': binding,
            'shape': list(spec.shape),
            'dtype': spec.dtype,
            'layout': spec.layout,
        })
    for idx, (binding, spec) in enumerate(zip(output_bindings, node.output_specs)):
        args.append({
            'kind': 'output',
            'index': idx,
            'binding': binding,
            'shape': list(spec.shape),
            'dtype': spec.dtype,
            'layout': spec.layout,
        })
    for idx, binding in enumerate(param_bindings):
        args.append({
            'kind': 'param',
            'index': idx,
            'binding': binding,
            'layout': param_layouts.get(binding, 'match_op_default'),
        })
    return tuple(args)


def _normalized_scalar_args(attr_bindings: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for key in sorted(attr_bindings):
        value = attr_bindings[key]
        normalized.append({
            'name': key,
            'value': value,
        })
    return tuple(normalized)


def build_gpu_dispatch_plan(graph: NativeGraph) -> GpuDispatchPlan:
    registry = {
        spec.op_name: spec
        for spec in list_gpu_kernel_specs()
    }
    lowering_specs = {
        spec.op_name: spec
        for spec in list_gpu_lowering_specs()
    }
    steps: list[GpuDispatchStep] = []
    unsupported_ops: list[str] = []
    for node in graph.topological_order():
        spec = registry.get(node.op_type)
        if spec is None:
            unsupported_ops.append(str(node.op_type))
            steps.append(
                GpuDispatchStep(
                    node_name=str(node.name),
                    op_name=str(node.op_type),
                    category='unsupported',
                    launch_family='unsupported',
                    input_names=tuple(str(name) for name in node.inputs),
                    output_names=tuple(str(name) for name in node.outputs),
                    param_keys=tuple(),
                    input_arity=len(node.inputs),
                    output_arity=len(node.outputs),
                    preferred_layout='unknown',
                    lowering_kind='unsupported',
                    launch_descriptor=GpuLaunchDescriptor(
                        launch_family='unsupported',
                        input_bindings=tuple(str(name) for name in node.inputs),
                        output_bindings=tuple(str(name) for name in node.outputs),
                        param_bindings=tuple(),
                        attr_bindings={},
                        input_shapes=tuple(tuple(spec.shape) for spec in node.input_specs),
                        output_shapes=tuple(tuple(spec.shape) for spec in node.output_specs),
                        tensor_dtype=str(node.output_specs[0].dtype if node.output_specs else 'float32'),
                        param_layouts={},
                        normalized_tensor_args=_normalized_tensor_args(
                            node,
                            tuple(str(name) for name in node.inputs),
                            tuple(str(name) for name in node.outputs),
                            tuple(),
                            {},
                        ),
                        normalized_scalar_args=tuple(),
                    ),
                    forward_status='unsupported',
                    backward_status='unsupported',
                    supported=False,
                )
            )
            continue
        lowering_kind = 'unbound'
        lowering_spec = lowering_specs.get(node.op_type)
        if lowering_spec is not None:
            lowering_kind = str(lowering_spec.lowering_kind)
        param_keys = _node_param_keys(node)
        input_bindings = tuple(str(name) for name in node.inputs)
        output_bindings = tuple(str(name) for name in node.outputs)
        attr_bindings = _node_attr_bindings(node)
        param_layouts = _node_param_layouts(node, param_keys)
        steps.append(
            GpuDispatchStep(
                node_name=str(node.name),
                op_name=str(node.op_type),
                category=str(spec.category),
                launch_family=str(spec.launch_family),
                input_names=input_bindings,
                output_names=output_bindings,
                param_keys=param_keys,
                input_arity=int(spec.input_arity),
                output_arity=int(spec.output_arity),
                preferred_layout=str(spec.preferred_layout),
                lowering_kind=lowering_kind,
                launch_descriptor=GpuLaunchDescriptor(
                    launch_family=str(spec.launch_family),
                    input_bindings=input_bindings,
                    output_bindings=output_bindings,
                    param_bindings=param_keys,
                    attr_bindings=attr_bindings,
                    input_shapes=tuple(tuple(spec.shape) for spec in node.input_specs),
                    output_shapes=tuple(tuple(spec.shape) for spec in node.output_specs),
                    tensor_dtype=str(node.output_specs[0].dtype if node.output_specs else 'float32'),
                    param_layouts=param_layouts,
                    normalized_tensor_args=_normalized_tensor_args(
                        node,
                        input_bindings,
                        output_bindings,
                        param_keys,
                        param_layouts,
                    ),
                    normalized_scalar_args=_normalized_scalar_args(attr_bindings),
                ),
                forward_status=str(spec.forward_status),
                backward_status=str(spec.backward_status),
                supported=True,
            )
        )
    return GpuDispatchPlan(
        execution_mode='gpu_native',
        ready=len(unsupported_ops) == 0,
        steps=tuple(steps),
        unsupported_ops=tuple(sorted(set(unsupported_ops))),
    )


def build_gpu_launch_packet(step: GpuDispatchStep) -> GpuLaunchPacket:
    return GpuLaunchPacket(
        node_name=step.node_name,
        op_name=step.op_name,
        launch_family=step.launch_family,
        lowering_kind=step.lowering_kind,
        preferred_layout=step.preferred_layout,
        tensor_args=tuple(dict(arg) for arg in step.launch_descriptor.normalized_tensor_args),
        scalar_args=tuple(dict(arg) for arg in step.launch_descriptor.normalized_scalar_args),
    )


def build_gpu_launch_trace(plan: GpuDispatchPlan) -> tuple[GpuLaunchPacket, ...]:
    return tuple(build_gpu_launch_packet(step) for step in plan.steps)
