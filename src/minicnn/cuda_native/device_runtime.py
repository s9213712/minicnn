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
    reservation_id: str | None = None
    device_ptr: Any | None = None
    owns_device_ptr: bool = False

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass
class DeviceRuntime:
    execution_mode: str = 'reference_numpy'
    tensor_execution_device: str = 'cpu'
    bound_lib: Any | None = None
    reserve_events: int = 0
    reserved_buffer_count: int = 0
    reserved_bytes: int = 0
    workspace_bytes: int = 0
    reserved_buffer_reuse_events: int = 0
    reserved_buffer_release_events: int = 0
    host_to_device_transfer_events: int = 0
    host_to_device_transfer_bytes: int = 0
    device_to_host_transfer_events: int = 0
    device_to_host_transfer_bytes: int = 0
    allocation_events: int = 0
    allocated_bytes: int = 0
    synchronization_events: int = 0
    synchronization_reasons: list[str] = field(default_factory=list)
    device_pointer_allocation_events: int = 0
    device_pointer_free_events: int = 0
    device_pointer_bytes: int = 0
    device_pointer_live_bytes: int = 0
    device_sync_to_host_events: int = 0
    device_sync_to_device_events: int = 0
    execution_events: int = 0
    executed_node_count: int = 0
    execution_kinds: dict[str, int] = field(default_factory=dict)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    last_input_name: str | None = None
    last_output_name: str | None = None
    _reserved_pool: dict[str, int] = field(default_factory=dict)
    _available_reserved_ids: list[str] = field(default_factory=list)

    @property
    def gpu_execution(self) -> bool:
        return self.tensor_execution_device == 'gpu'

    @property
    def native_device_pointers_enabled(self) -> bool:
        return self.gpu_execution and self.bound_lib is not None

    def _malloc_device(self, nbytes: int) -> Any | None:
        if not self.native_device_pointers_enabled:
            return None
        ptr = self.bound_lib.gpu_malloc(int(nbytes))
        self.device_pointer_allocation_events += 1
        self.device_pointer_bytes += int(nbytes)
        self.device_pointer_live_bytes += int(nbytes)
        return ptr

    def _free_device(self, tensor: DeviceTensor) -> None:
        if tensor.device_ptr is None or not tensor.owns_device_ptr or self.bound_lib is None:
            return
        self.bound_lib.gpu_free(tensor.device_ptr)
        self.device_pointer_free_events += 1
        self.device_pointer_live_bytes -= int(tensor.nbytes)
        tensor.device_ptr = None
        tensor.owns_device_ptr = False

    def sync_tensor_to_device(self, tensor: DeviceTensor) -> None:
        if tensor.device_ptr is None or self.bound_lib is None:
            return
        arr = np.ascontiguousarray(tensor.data)
        if arr is not tensor.data:
            tensor.data = arr
        self.bound_lib.gpu_memcpy_h2d(tensor.device_ptr, tensor.data.ctypes.data, int(tensor.data.nbytes))
        self.device_sync_to_device_events += 1

    def sync_tensor_to_host(self, tensor: DeviceTensor) -> None:
        if tensor.device_ptr is None or self.bound_lib is None:
            return
        self.bound_lib.gpu_memcpy_d2h(tensor.data.ctypes.data, tensor.device_ptr, int(tensor.data.nbytes))
        self.device_sync_to_host_events += 1

    def stage_to_device(
        self,
        array: Any,
        *,
        name: str | None = None,
        copy: bool = True,
        prefer_reserved: bool = False,
    ) -> DeviceTensor:
        host_array = np.asarray(array)
        self.host_to_device_transfer_events += 1
        self.host_to_device_transfer_bytes += int(host_array.nbytes)
        if prefer_reserved:
            staged = self.allocate_staging_buffer(host_array.shape, dtype=host_array.dtype, name=name)
            np.copyto(staged.data, host_array)
            self.sync_tensor_to_device(staged)
            return staged
        if copy:
            host_array = np.array(host_array, copy=True)
        tensor = DeviceTensor(
            host_array,
            self.tensor_execution_device,
            self.execution_mode,
            name=name,
            device_ptr=self._malloc_device(int(host_array.nbytes)),
            owns_device_ptr=self.native_device_pointers_enabled,
        )
        self.sync_tensor_to_device(tensor)
        return tensor

    def stage_to_host(self, tensor: DeviceTensor, *, copy: bool = True) -> np.ndarray:
        self.sync_tensor_to_host(tensor)
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
        return DeviceTensor(
            array,
            self.tensor_execution_device,
            self.execution_mode,
            name=name,
            device_ptr=self._malloc_device(int(array.nbytes)),
            owns_device_ptr=self.native_device_pointers_enabled,
        )

    def reserve_from_planner(
        self,
        *,
        total_bytes: int,
        num_buffers: int,
        workspace_bytes: int = 0,
        buffer_capacities: dict[str, int] | None = None,
    ) -> None:
        self.reserve_events += 1
        self.reserved_buffer_count = int(num_buffers)
        self.reserved_bytes = int(total_bytes)
        self.workspace_bytes = int(workspace_bytes)
        self._reserved_pool.clear()
        self._available_reserved_ids.clear()
        if buffer_capacities:
            for buffer_id, capacity in buffer_capacities.items():
                buffer_name = str(buffer_id)
                self._reserved_pool[buffer_name] = int(capacity)
                self._available_reserved_ids.append(buffer_name)
        else:
            avg_capacity = int(total_bytes // max(num_buffers, 1)) if num_buffers > 0 else 0
            for idx in range(int(num_buffers)):
                buffer_name = f'reserved_{idx}'
                self._reserved_pool[buffer_name] = avg_capacity
                self._available_reserved_ids.append(buffer_name)

    def allocate_staging_buffer(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str | np.dtype = 'float32',
        name: str | None = None,
    ) -> DeviceTensor:
        dtype_obj = np.dtype(dtype)
        required_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype_obj.itemsize)
        chosen_id = None
        for idx, buffer_id in enumerate(self._available_reserved_ids):
            if int(self._reserved_pool.get(buffer_id, 0)) >= required_bytes:
                chosen_id = self._available_reserved_ids.pop(idx)
                break
        array = np.zeros(shape, dtype=dtype_obj)
        device_ptr = self._malloc_device(required_bytes)
        owns_device_ptr = self.native_device_pointers_enabled
        if chosen_id is not None:
            self.reserved_buffer_reuse_events += 1
            return DeviceTensor(
                array,
                self.tensor_execution_device,
                self.execution_mode,
                name=name,
                reservation_id=chosen_id,
                device_ptr=device_ptr,
                owns_device_ptr=owns_device_ptr,
            )
        self.allocation_events += 1
        self.allocated_bytes += int(array.nbytes)
        return DeviceTensor(
            array,
            self.tensor_execution_device,
            self.execution_mode,
            name=name,
            device_ptr=device_ptr,
            owns_device_ptr=owns_device_ptr,
        )

    def release_buffer(self, tensor: DeviceTensor) -> None:
        self._free_device(tensor)
        if tensor.reservation_id is None:
            return
        self.reserved_buffer_release_events += 1
        self._available_reserved_ids.append(tensor.reservation_id)

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
        self.execution_trace.append(
            {
                'kind': execution_kind,
                'input_name': input_name,
                'output_name': output_name,
                'node_count': int(node_count),
            }
        )
        self.last_input_name = input_name
        self.last_output_name = output_name

    def summary(self) -> dict[str, Any]:
        return {
            'execution_mode': self.execution_mode,
            'effective_execution_mode': self.execution_mode,
            'tensor_execution_device': self.tensor_execution_device,
            'tensors_ran_on': self.tensor_execution_device,
            'gpu_execution': self.gpu_execution,
            'native_device_pointers_enabled': self.native_device_pointers_enabled,
            'reserve_events': self.reserve_events,
            'reserved_buffer_count': self.reserved_buffer_count,
            'reserved_bytes': self.reserved_bytes,
            'workspace_bytes': self.workspace_bytes,
            'reserved_buffer_reuse_events': self.reserved_buffer_reuse_events,
            'reserved_buffer_release_events': self.reserved_buffer_release_events,
            'available_reserved_buffer_count': len(self._available_reserved_ids),
            'host_to_device_transfer_events': self.host_to_device_transfer_events,
            'host_to_device_transfer_bytes': self.host_to_device_transfer_bytes,
            'device_to_host_transfer_events': self.device_to_host_transfer_events,
            'device_to_host_transfer_bytes': self.device_to_host_transfer_bytes,
            'allocation_events': self.allocation_events,
            'allocated_bytes': self.allocated_bytes,
            'synchronization_events': self.synchronization_events,
            'synchronization_reasons': list(self.synchronization_reasons),
            'device_pointer_allocation_events': self.device_pointer_allocation_events,
            'device_pointer_free_events': self.device_pointer_free_events,
            'device_pointer_bytes': self.device_pointer_bytes,
            'device_pointer_live_bytes': self.device_pointer_live_bytes,
            'device_sync_to_device_events': self.device_sync_to_device_events,
            'device_sync_to_host_events': self.device_sync_to_host_events,
            'execution_events': self.execution_events,
            'executed_node_count': self.executed_node_count,
            'execution_kinds': dict(self.execution_kinds),
            'execution_trace': [dict(item) for item in self.execution_trace],
            'last_input_name': self.last_input_name,
            'last_output_name': self.last_output_name,
        }
