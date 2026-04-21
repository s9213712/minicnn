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
]
