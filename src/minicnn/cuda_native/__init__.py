"""cuda_native — experimental sequential-only forward-only native CUDA backend.

Status: experimental
Training: not supported
Backward: not supported
Graph mode: sequential only
"""
from minicnn.cuda_native.capabilities import CUDA_NATIVE_CAPABILITIES, get_cuda_native_capabilities
from minicnn.cuda_native.api import (
    validate_cuda_native_config,
    build_cuda_native_graph,
    get_capability_summary,
)
from minicnn.cuda_native.graph import NativeGraph, build_graph
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.planner import ExecutionPlan, BufferPlan, make_naive_plan
from minicnn.cuda_native.backward import BackwardExecutor, make_default_backward_registry
from minicnn.cuda_native.loss import cross_entropy_loss, mse_loss
from minicnn.cuda_native.training import train_step, sgd_update
from minicnn.cuda_native.layouts import (
    NCHW, NHWC, NC, C, SCALAR,
    LayoutSpec, infer_layout,
    validate_op_layout, validate_graph_layouts,
    OP_LAYOUT_CONTRACT,
)
from minicnn.cuda_native.memory import (
    BufferAllocator, BufferPool, memory_footprint,
)
from minicnn.cuda_native.debug import (
    dump_graph, print_graph,
    dump_plan, print_plan,
    inspect, print_inspect,
    ExecutionTrace, NodeTrace, TracingForwardExecutor,
)

__all__ = [
    'CUDA_NATIVE_CAPABILITIES',
    'get_cuda_native_capabilities',
    'validate_cuda_native_config',
    'build_cuda_native_graph',
    'get_capability_summary',
    'NativeGraph',
    'build_graph',
    'ForwardExecutor',
    'ExecutionPlan',
    'BufferPlan',
    'make_naive_plan',
    'BackwardExecutor',
    'make_default_backward_registry',
    'cross_entropy_loss',
    'mse_loss',
    'train_step',
    'sgd_update',
    'dump_graph',
    'print_graph',
    'dump_plan',
    'print_plan',
    'inspect',
    'print_inspect',
    'ExecutionTrace',
    'NodeTrace',
    'TracingForwardExecutor',
    'NCHW', 'NHWC', 'NC', 'C', 'SCALAR',
    'LayoutSpec', 'infer_layout',
    'validate_op_layout', 'validate_graph_layouts',
    'OP_LAYOUT_CONTRACT',
    'BufferAllocator', 'BufferPool', 'memory_footprint',
]
