from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime


def test_device_runtime_tracks_staging_allocation_and_sync():
    runtime = DeviceRuntime(execution_mode='reference_numpy', tensor_execution_device='cpu')
    runtime.reserve_from_planner(total_bytes=256, num_buffers=3, workspace_bytes=64)

    tensor = runtime.stage_to_device(np.ones((2, 3), dtype=np.float32), name='x')
    host = runtime.stage_to_host(tensor)
    allocated = runtime.allocate((2, 3), dtype='float32', name='buf')
    staged_output = runtime.allocate_staging_buffer((2, 3), dtype='float32', name='y')
    runtime.record_execution('eval_forward', input_name='x', output_name='y', node_count=4)
    runtime.release_buffer(staged_output)
    runtime.synchronize('test-barrier')

    assert tensor.execution_mode == 'reference_numpy'
    assert tensor.device == 'cpu'
    assert host.shape == (2, 3)
    assert allocated.nbytes == 24

    summary = runtime.summary()
    assert summary['execution_mode'] == 'reference_numpy'
    assert summary['tensor_execution_device'] == 'cpu'
    assert summary['gpu_execution'] is False
    assert summary['reserve_events'] == 1
    assert summary['reserved_buffer_count'] == 3
    assert summary['reserved_bytes'] == 256
    assert summary['workspace_bytes'] == 64
    assert summary['reserved_buffer_reuse_events'] == 1
    assert summary['reserved_buffer_release_events'] == 1
    assert summary['available_reserved_buffer_count'] == 3
    assert summary['host_to_device_transfer_events'] == 1
    assert summary['device_to_host_transfer_events'] == 1
    assert summary['allocation_events'] == 1
    assert summary['allocated_bytes'] == 24
    assert summary['execution_events'] == 1
    assert summary['executed_node_count'] == 4
    assert summary['execution_kinds'] == {'eval_forward': 1}
    assert summary['last_input_name'] == 'x'
    assert summary['last_output_name'] == 'y'
    assert summary['synchronization_events'] == 1
    assert summary['synchronization_reasons'] == ['test-barrier']
