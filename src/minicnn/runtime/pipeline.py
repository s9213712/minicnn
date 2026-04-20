"""
Connects the compiler pipeline to the runtime executor.

Typical usage:

    from minicnn.runtime.pipeline import InferencePipeline
    pipeline = InferencePipeline.from_config(model_cfg)
    output_values = pipeline.run(x)          # dict of {node_name: Tensor}
    logits = pipeline.run_final(x)           # Tensor of the last node
"""
from __future__ import annotations

from minicnn.compiler.ir import IRGraph
from minicnn.compiler.optimizer import optimize
from minicnn.compiler.tracer import trace_model_config
from minicnn.runtime.executor import GraphExecutor
from minicnn.runtime.graph import Graph, Node
from minicnn.runtime.profiler import Profiler


def ir_to_runtime_graph(ir_graph: IRGraph) -> Graph:
    """Convert a compiler IRGraph to a runtime Graph for GraphExecutor."""
    g = Graph()
    for node in ir_graph.nodes:
        g.add_node(Node(name=node.name, op=node.op, inputs=node.inputs, attrs=node.attrs))
    return g


class InferencePipeline:
    """End-to-end inference: YAML model config → optimized IR → execution."""

    def __init__(self, executor: GraphExecutor, profiler: Profiler | None = None):
        self._executor = executor
        self._profiler = profiler or Profiler()

    @classmethod
    def from_config(cls, model_cfg: dict, *, profile: bool = False) -> 'InferencePipeline':
        ir = trace_model_config(model_cfg)
        ir = optimize(ir)
        graph = ir_to_runtime_graph(ir)
        executor = GraphExecutor(graph)
        return cls(executor, Profiler() if profile else None)

    @classmethod
    def from_ir(cls, ir_graph: IRGraph) -> 'InferencePipeline':
        graph = ir_to_runtime_graph(ir_graph)
        return cls(GraphExecutor(graph))

    def run(self, x) -> dict:
        """Run the full graph; returns a dict mapping node name → output value."""
        with self._profiler.record('forward'):
            return self._executor.run(x)

    def run_final(self, x):
        """Run the graph and return only the last node's output."""
        values = self.run(x)
        last_node = list(self._executor.graph.nodes)[-1]
        return values[last_node]

    def profile_summary(self) -> dict:
        return self._profiler.summary()
