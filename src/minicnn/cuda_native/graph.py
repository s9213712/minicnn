"""Ordered graph container for cuda_native."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minicnn.cuda_native.nodes import Node, TensorSpec
from minicnn.cuda_native.shapes import infer_shape
from minicnn.cuda_native.validators import validate_layer_attrs, validate_op_type


@dataclass
class NativeGraph:
    """Immutable-ish ordered graph produced by build_graph()."""
    nodes: list[Node] = field(default_factory=list)
    input_spec: TensorSpec | None = None
    output_spec: TensorSpec | None = None

    def topological_order(self) -> list[Node]:
        """For ordered DAGs this is the insertion order established at build time."""
        return list(self.nodes)

    def summary(self) -> dict[str, object]:
        return {
            'num_nodes': len(self.nodes),
            'ops': [n.op_type for n in self.nodes],
            'input_shape': self.input_spec.shape if self.input_spec else None,
            'output_shape': self.output_spec.shape if self.output_spec else None,
        }

    def __repr__(self) -> str:
        lines = ['NativeGraph(']
        for n in self.nodes:
            lines.append(f'  {n}')
        lines.append(')')
        return '\n'.join(lines)


def build_graph(
    layers: list[dict[str, Any]],
    input_shape: tuple[int, ...],
) -> NativeGraph:
    """Build a NativeGraph from a list of layer dicts and a fixed input shape.

    Performs validation and shape inference in one pass.

    Args:
        layers: list of dicts like [{'type': 'Conv2d', 'out_channels': 32}, ...]
        input_shape: (N, C, H, W) or (N, features) for the first layer

    Returns:
        A fully-annotated NativeGraph with TensorSpec on every node.

    Raises:
        ValueError: on unsupported op, missing attrs, or illegal shapes.
    """
    errors: list[str] = []
    for i, layer in enumerate(layers):
        op = str(layer.get('type', ''))
        node_name = str(layer.get('name') or f'{op.lower()}_{i}')
        op_errors = validate_op_type(op, node_name=node_name)
        errors.extend(op_errors)
        if not op_errors:
            attrs = {k: v for k, v in layer.items() if k != 'type'}
            errors.extend(validate_layer_attrs(op, attrs, node_name))
    if errors:
        raise ValueError('cuda_native validation failed:\n- ' + '\n- '.join(errors))

    graph = NativeGraph()
    input_tensor = TensorSpec(name='input', shape=tuple(input_shape))
    graph.input_spec = input_tensor
    tensor_specs: dict[str, TensorSpec] = {'input': input_tensor}
    prev_name = 'input'

    for i, layer in enumerate(layers):
        op = str(layer['type'])
        node_name = str(layer.get('name') or f'{op.lower()}_{i}')
        attrs = {
            k: v for k, v in layer.items() if k not in {'type', 'name', 'inputs', 'output'}
        }
        input_names_raw = layer.get('inputs')
        if input_names_raw is None:
            input_names = [prev_name]
        elif isinstance(input_names_raw, list):
            input_names = [str(name) for name in input_names_raw]
        else:
            raise ValueError(
                f'{op} node={node_name}: attr "inputs" must be a list of tensor names'
            )

        input_specs: list[TensorSpec] = []
        for input_name in input_names:
            if input_name not in tensor_specs:
                raise ValueError(
                    f'{op} node={node_name}: unknown input tensor {input_name!r}. '
                    f'Known tensors: {sorted(tensor_specs)}'
                )
            spec = tensor_specs[input_name]
            input_specs.append(
                TensorSpec(
                    name=spec.name,
                    shape=spec.shape,
                    dtype=spec.dtype,
                    layout=spec.layout,
                )
            )

        input_shapes = [spec.shape for spec in input_specs]
        out_shape = infer_shape(
            op,
            input_shapes if len(input_shapes) > 1 else input_shapes[0],
            attrs,
            node_name=node_name,
        )

        out_name = str(layer.get('output', f't_{i + 1}'))
        if out_name in tensor_specs:
            raise ValueError(
                f'{op} node={node_name}: output tensor {out_name!r} already exists. '
                'Choose a unique output name.'
            )
        out_spec = TensorSpec(name=out_name, shape=out_shape)
        tensor_specs[out_name] = out_spec
        prev_name = out_name

        node = Node(
            name=node_name,
            op_type=op,
            inputs=[spec.name for spec in input_specs],
            outputs=[out_spec.name],
            attrs=attrs,
            input_specs=input_specs,
            output_specs=[out_spec],
        )
        graph.nodes.append(node)

    graph.output_spec = tensor_specs[prev_name]
    return graph
