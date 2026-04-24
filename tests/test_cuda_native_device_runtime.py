from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime


def test_device_runtime_tracks_staging_allocation_and_sync():
    runtime = DeviceRuntime(execution_mode='reference_numpy', tensor_execution_device='cpu')

    tensor = runtime.stage_to_device(np.ones((2, 3), dtype=np.float32), name='x')
    host = runtime.stage_to_host(tensor)
    allocated = runtime.allocate((2, 3), dtype='float32', name='buf')
    runtime.synchronize('test-barrier')

    assert tensor.execution_mode == 'reference_numpy'
    assert tensor.device == 'cpu'
    assert host.shape == (2, 3)
    assert allocated.nbytes == 24

    summary = runtime.summary()
    assert summary['execution_mode'] == 'reference_numpy'
    assert summary['tensor_execution_device'] == 'cpu'
    assert summary['gpu_execution'] is False
    assert summary['host_to_device_transfer_events'] == 1
    assert summary['device_to_host_transfer_events'] == 1
    assert summary['allocation_events'] == 1
    assert summary['allocated_bytes'] == 24
    assert summary['synchronization_events'] == 1
    assert summary['synchronization_reasons'] == ['test-barrier']
