from __future__ import annotations


def test_cuda_environment_diagnostics_reports_wsl_device_node_and_runtime_mismatch(monkeypatch):
    import minicnn.core.cuda_backend as cuda_backend

    monkeypatch.setattr(cuda_backend, '_is_wsl_environment', lambda: True)
    monkeypatch.setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64:')

    def exists(path: str) -> bool:
        return path in {
            '/usr/lib/wsl/lib/libcuda.so.1',
            '/lib/x86_64-linux-gnu/libcuda.so.1',
        }

    diagnostics = cuda_backend._cuda_environment_diagnostics(
        {
            'status': 35,
            'driver_version': 0,
            'runtime_version': 13020,
        },
        exists_fn=exists,
        realpath_fn=lambda path: f'/resolved{path}',
    )

    assert diagnostics['wsl'] is True
    assert diagnostics['device_nodes']['/dev/dxg'] is False
    assert diagnostics['device_nodes']['/dev/nvidiactl'] is False
    assert diagnostics['runtime_driver_mismatch'] is True
    assert diagnostics['issue'] == 'wsl_cuda_device_node_missing'
    assert diagnostics['ld_library_path'] == '/usr/local/cuda/lib64:'
    assert diagnostics['libcuda_candidates'][0] == {
        'path': '/usr/lib/wsl/lib/libcuda.so.1',
        'exists': True,
        'resolved_path': '/resolved/usr/lib/wsl/lib/libcuda.so.1',
    }
    assert any('/dev/dxg' in item for item in diagnostics['remediation'])
    assert any('driver=unknown, runtime=13.2' in item for item in diagnostics['remediation'])


def test_cuda_environment_diagnostics_reports_driver_runtime_mismatch_without_wsl(monkeypatch):
    import minicnn.core.cuda_backend as cuda_backend

    monkeypatch.setattr(cuda_backend, '_is_wsl_environment', lambda: False)

    diagnostics = cuda_backend._cuda_environment_diagnostics(
        {
            'status': 35,
            'driver_version': 12020,
            'runtime_version': 13020,
        },
        exists_fn=lambda path: False,
    )

    assert diagnostics['wsl'] is False
    assert diagnostics['runtime_driver_mismatch'] is True
    assert diagnostics['issue'] == 'cuda_driver_runtime_mismatch'
    assert diagnostics['remediation'] == [
        'Use a CUDA runtime/toolkit compatible with the driver visible to the process '
        '(driver=12.2, runtime=13.2).'
    ]
