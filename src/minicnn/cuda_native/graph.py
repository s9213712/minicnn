"""Sequential graph container for cuda_native."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minicnn.cuda_native.nodes import Node, TensorSpec
from minicnn.cuda_native.shapes import infer_shape
from minicnn.cuda_native.validators import validate_op_type


@dataclass
class NativeGraph:
    """Immutable-ish sequential graph produced by build_graph()."""
    nodes: list[Node] = field(default_factory=list)
    input_spec: TensorSpec | None = None
    output_spec: TensorSpec | None = None

    def topological_order(self) -> list[Node]:
        """For sequential graphs this is just insertion order."""
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
        errors.extend(validate_op_type(op, node_name=f'layer_{i}'))
    if errors:
        raise ValueError('cuda_native validation failed:\n- ' + '\n- '.join(errors))

    graph = NativeGraph()
    current_shape = tuple(input_shape)
    input_tensor = TensorSpec(name='input', shape=current_shape)
    graph.input_spec = input_tensor

    # Tensor naming: 'input' for the feed, then 't_1', 't_2', ... for intermediates.
    # This ensures node.inputs[0] == 'input' for the first node, matching the feed dict.
    prev_name = 'input'

    for i, layer in enumerate(layers):
        op = str(layer['type'])
        attrs = {k: v for k, v in layer.items() if k != 'type'}
        node_name = f'{op.lower()}_{i}'

        out_shape = infer_shape(op, current_shape, attrs, node_name=node_name)

        in_spec = TensorSpec(name=prev_name, shape=current_shape)
        out_name = f't_{i + 1}'
        out_spec = TensorSpec(name=out_name, shape=out_shape)
        prev_name = out_name

        node = Node(
            name=node_name,
            op_type=op,
            inputs=[in_spec.name],
            outputs=[out_spec.name],
            attrs=attrs,
            input_specs=[in_spec],
            output_specs=[out_spec],
        )
        graph.nodes.append(node)
        current_shape = out_shape

    # output_spec.name matches the last node's output tensor name
    graph.output_spec = TensorSpec(name=prev_name, shape=current_shape)
    return graph
