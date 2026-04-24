from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DeviceTensor:
    data: np.ndarray
    device: str
    execution_mode: str
    name: str | None = None

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass
class DeviceRuntime:
    execution_mode: str = 'reference_numpy'
    tensor_execution_device: str = 'cpu'
    reserve_events: int = 0
    reserved_buffer_count: int = 0
    reserved_bytes: int = 0
    workspace_bytes: int = 0
    host_to_device_transfer_events: int = 0
    host_to_device_transfer_bytes: int = 0
    device_to_host_transfer_events: int = 0
    device_to_host_transfer_bytes: int = 0
    allocation_events: int = 0
    allocated_bytes: int = 0
    synchronization_events: int = 0
    synchronization_reasons: list[str] = field(default_factory=list)
    execution_events: int = 0
    executed_node_count: int = 0
    execution_kinds: dict[str, int] = field(default_factory=dict)
    last_input_name: str | None = None
    last_output_name: str | None = None

    @property
    def gpu_execution(self) -> bool:
        return self.tensor_execution_device == 'gpu'

    def stage_to_device(self, array: Any, *, name: str | None = None, copy: bool = True) -> DeviceTensor:
        host_array = np.asarray(array)
        if copy:
            host_array = np.array(host_array, copy=True)
        self.host_to_device_transfer_events += 1
        self.host_to_device_transfer_bytes += int(host_array.nbytes)
        return DeviceTensor(host_array, self.tensor_execution_device, self.execution_mode, name=name)

    def stage_to_host(self, tensor: DeviceTensor, *, copy: bool = True) -> np.ndarray:
        host_array = tensor.data
        if copy:
            host_array = np.array(host_array, copy=True)
        self.device_to_host_transfer_events += 1
        self.device_to_host_transfer_bytes += int(host_array.nbytes)
        return host_array

    def allocate(self, shape: tuple[int, ...], *, dtype: str | np.dtype = 'float32', name: str | None = None) -> DeviceTensor:
        array = np.zeros(shape, dtype=np.dtype(dtype))
        self.allocation_events += 1
        self.allocated_bytes += int(array.nbytes)
        return DeviceTensor(array, self.tensor_execution_device, self.execution_mode, name=name)

    def reserve_from_planner(
        self,
        *,
        total_bytes: int,
        num_buffers: int,
        workspace_bytes: int = 0,
    ) -> None:
        self.reserve_events += 1
        self.reserved_buffer_count = int(num_buffers)
        self.reserved_bytes = int(total_bytes)
        self.workspace_bytes = int(workspace_bytes)

    def synchronize(self, reason: str = 'explicit') -> None:
        self.synchronization_events += 1
        self.synchronization_reasons.append(str(reason))

    def record_execution(
        self,
        kind: str,
        *,
        input_name: str | None = None,
        output_name: str | None = None,
        node_count: int = 0,
    ) -> None:
        execution_kind = str(kind)
        self.execution_events += 1
        self.executed_node_count += int(node_count)
        self.execution_kinds[execution_kind] = int(self.execution_kinds.get(execution_kind, 0)) + 1
        self.last_input_name = input_name
        self.last_output_name = output_name

    def summary(self) -> dict[str, Any]:
        return {
            'execution_mode': self.execution_mode,
            'effective_execution_mode': self.execution_mode,
            'tensor_execution_device': self.tensor_execution_device,
            'tensors_ran_on': self.tensor_execution_device,
            'gpu_execution': self.gpu_execution,
            'reserve_events': self.reserve_events,
            'reserved_buffer_count': self.reserved_buffer_count,
            'reserved_bytes': self.reserved_bytes,
            'workspace_bytes': self.workspace_bytes,
            'host_to_device_transfer_events': self.host_to_device_transfer_events,
            'host_to_device_transfer_bytes': self.host_to_device_transfer_bytes,
            'device_to_host_transfer_events': self.device_to_host_transfer_events,
            'device_to_host_transfer_bytes': self.device_to_host_transfer_bytes,
            'allocation_events': self.allocation_events,
            'allocated_bytes': self.allocated_bytes,
            'synchronization_events': self.synchronization_events,
            'synchronization_reasons': list(self.synchronization_reasons),
            'execution_events': self.execution_events,
            'executed_node_count': self.executed_node_count,
            'execution_kinds': dict(self.execution_kinds),
            'last_input_name': self.last_input_name,
            'last_output_name': self.last_output_name,
        }
