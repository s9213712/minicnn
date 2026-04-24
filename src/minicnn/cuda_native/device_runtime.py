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
    host_to_device_transfer_events: int = 0
    host_to_device_transfer_bytes: int = 0
    device_to_host_transfer_events: int = 0
    device_to_host_transfer_bytes: int = 0
    allocation_events: int = 0
    allocated_bytes: int = 0
    synchronization_events: int = 0
    synchronization_reasons: list[str] = field(default_factory=list)

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

    def synchronize(self, reason: str = 'explicit') -> None:
        self.synchronization_events += 1
        self.synchronization_reasons.append(str(reason))

    def summary(self) -> dict[str, Any]:
        return {
            'execution_mode': self.execution_mode,
            'effective_execution_mode': self.execution_mode,
            'tensor_execution_device': self.tensor_execution_device,
            'tensors_ran_on': self.tensor_execution_device,
            'gpu_execution': self.gpu_execution,
            'host_to_device_transfer_events': self.host_to_device_transfer_events,
            'host_to_device_transfer_bytes': self.host_to_device_transfer_bytes,
            'device_to_host_transfer_events': self.device_to_host_transfer_events,
            'device_to_host_transfer_bytes': self.device_to_host_transfer_bytes,
            'allocation_events': self.allocation_events,
            'allocated_bytes': self.allocated_bytes,
            'synchronization_events': self.synchronization_events,
            'synchronization_reasons': list(self.synchronization_reasons),
        }
