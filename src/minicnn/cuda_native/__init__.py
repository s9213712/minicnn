"""cuda_native — beta ordered-graph native backend.

Status: beta
Training: beta-grade within the NumPy-reference execution model
Backward: beta-grade within the NumPy-reference execution model
Graph mode: ordered DAG
"""
from minicnn.cuda_native.capabilities import CUDA_NATIVE_CAPABILITIES, get_cuda_native_capabilities
from minicnn.cuda_native.api import (
    validate_cuda_native_config,
    build_cuda_native_graph,
    get_capability_summary,
)
from minicnn.cuda_native.device_runtime import DeviceRuntime, DeviceTensor
from minicnn.cuda_native.device_runtime import DeviceRuntime, DeviceTensor
from minicnn.cuda_native.graph import NativeGraph, build_graph
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.gpu_executor import GpuStubExecutor, make_native_gpu_forward_executor
from minicnn.cuda_native.gpu_training import (
    NativeGpuConvLinearTrainingStepResult,
    NativeGpuLinearTrainingStepResult,
    NativeGpuPoolLinearTrainingStepResult,
    NativeGpuTwoConvReluPoolLinearTrainingStepResult,
    NativeGpuTwoLinearReluTrainingStepResult,
    native_gpu_avgpool_linear_training_step,
    native_gpu_conv_linear_training_step,
    native_gpu_linear_training_step,
    native_gpu_pool_linear_training_step,
    native_gpu_two_conv_relu_pool_linear_training_step,
    native_gpu_two_linear_relu_training_step,
)
from minicnn.cuda_native.planner import (
    BufferPlan,
    BufferType,
    ExecutionPlan,
    analyze_live_tensor_sets,
    estimate_peak_live_bytes,
    analyze_live_ranges,
    make_naive_plan,
    make_plan,
    make_reuse_plan,
)
from minicnn.cuda_native.backward import BackwardExecutor, make_default_backward_registry
from minicnn.cuda_native.loss import cross_entropy_loss, mse_loss
from minicnn.cuda_native.training import train_step, sgd_update
from minicnn.cuda_native.layouts import (
    NCHW, NHWC, NC, C, SCALAR,
    LayoutSpec, infer_layout,
    validate_op_layout, validate_graph_layouts,
    OP_LAYOUT_RULES,
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
    'GpuStubExecutor',
    'make_native_gpu_forward_executor',
    'NativeGpuConvLinearTrainingStepResult',
    'NativeGpuLinearTrainingStepResult',
    'NativeGpuPoolLinearTrainingStepResult',
    'NativeGpuTwoConvReluPoolLinearTrainingStepResult',
    'NativeGpuTwoLinearReluTrainingStepResult',
    'native_gpu_avgpool_linear_training_step',
    'native_gpu_conv_linear_training_step',
    'native_gpu_linear_training_step',
    'native_gpu_pool_linear_training_step',
    'native_gpu_two_conv_relu_pool_linear_training_step',
    'native_gpu_two_linear_relu_training_step',
    'ExecutionPlan',
    'BufferPlan',
    'BufferType',
    'analyze_live_tensor_sets',
    'estimate_peak_live_bytes',
    'analyze_live_ranges',
    'make_naive_plan',
    'make_reuse_plan',
    'make_plan',
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
    'OP_LAYOUT_RULES',
    'BufferAllocator', 'BufferPool', 'memory_footprint',
    'DeviceRuntime', 'DeviceTensor',
]
