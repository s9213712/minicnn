from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any

from minicnn.cuda_native.gpu_bridge import GpuCAbiKernelCall, GpuFixedKernelCall, GpuFlatKernelRequest, GpuKernelBridgeRequest


class GpuKernelBridgeAdapter(Protocol):
    def submit(self, request: GpuKernelBridgeRequest) -> dict[str, Any]:
        ...


@dataclass
class GpuStubBridgeAdapter:
    submitted_requests: list[GpuKernelBridgeRequest] = field(default_factory=list)

    def submit(self, request: GpuKernelBridgeRequest) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': request.dispatch_mode,
            'launch_family': request.launch_family,
            'accepted': True,
        }


@dataclass
class GpuFlatBridgeAdapter:
    submitted_requests: list[GpuFlatKernelRequest] = field(default_factory=list)

    def submit_flat(self, request: GpuFlatKernelRequest) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': request.dispatch_mode,
            'launch_family': request.launch_family,
            'flat_tensor_arg_count': len(request.tensor_bindings),
            'flat_scalar_arg_count': len(request.scalar_names),
            'accepted': True,
        }


@dataclass
class GpuFixedBridgeAdapter:
    submitted_requests: list[GpuFixedKernelCall] = field(default_factory=list)

    def submit_fixed(self, request: GpuFixedKernelCall) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': request.dispatch_mode,
            'launch_family': request.launch_family,
            'has_weight_binding': bool(request.weight_binding),
            'matmul_signature': [request.matmul_m, request.matmul_k, request.matmul_n],
            'accepted': True,
        }


@dataclass
class GpuBackendStubAdapter:
    submitted_requests: list[GpuFixedKernelCall] = field(default_factory=list)
    abi_version: int = 1

    def submit_fixed(self, request: GpuFixedKernelCall) -> dict[str, Any]:
        self.submitted_requests.append(request)
        if request.launch_family == 'gemm_affine':
            return self._submit_linear(request)
        if request.launch_family == 'conv2d_nchw':
            return self._submit_conv2d(request)
        return self._submit_generic(request)

    def _base_result(self, request: GpuFixedKernelCall, *, kernel_symbol: str) -> dict[str, Any]:
        return {
            'request_id': request.request_id,
            'dispatch_mode': 'gpu_backend_stub',
            'abi_version': self.abi_version,
            'launch_family': request.launch_family,
            'kernel_symbol': kernel_symbol,
            'accepted': True,
        }

    def _submit_linear(self, request: GpuFixedKernelCall) -> dict[str, Any]:
        result = self._base_result(request, kernel_symbol='minicnn_gpu_linear_f32')
        result.update({
            'matmul_signature': [request.matmul_m, request.matmul_k, request.matmul_n],
            'input_binding': request.input_binding,
            'output_binding': request.output_binding,
            'weight_binding': request.weight_binding,
            'bias_binding': request.bias_binding,
        })
        return result

    def _submit_conv2d(self, request: GpuFixedKernelCall) -> dict[str, Any]:
        result = self._base_result(request, kernel_symbol='minicnn_gpu_conv2d_nchw_f32')
        result.update({
            'input_binding': request.input_binding,
            'output_binding': request.output_binding,
            'weight_binding': request.weight_binding,
            'bias_binding': request.bias_binding,
            'input_shape': list(request.input_shape),
            'output_shape': list(request.output_shape),
            'stride': [request.stride_h, request.stride_w],
            'padding': [request.padding_h, request.padding_w],
            'groups': request.groups,
        })
        return result

    def _submit_generic(self, request: GpuFixedKernelCall) -> dict[str, Any]:
        return self._base_result(
            request,
            kernel_symbol=f"minicnn_gpu_{request.launch_family}_f32",
        )


@dataclass
class GpuCAbiBridgeAdapter:
    submitted_requests: list[GpuCAbiKernelCall] = field(default_factory=list)
    abi_version: int = 1

    def submit_c_abi(self, request: GpuCAbiKernelCall) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': 'gpu_c_abi_stub',
            'abi_version': self.abi_version,
            'op_code': request.op_code,
            'launch_family_code': request.launch_family_code,
            'dtype_code': request.dtype_code,
            'int_args8': list(request.int_args8),
            'accepted': True,
        }
