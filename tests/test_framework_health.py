from __future__ import annotations

from pathlib import Path


def test_find_native_artifacts_recognizes_cross_platform_libraries(tmp_path: Path):
    from minicnn.framework.health import find_native_artifacts

    for name in (
        'libminimal_cuda_cnn.so',
        'minimal_cuda_cnn_handmade.dll',
        'libminimal_cuda_cnn.dylib',
        'minimal_cuda_cnn.lib',
        'README.md',
    ):
        (tmp_path / name).write_text('', encoding='utf-8')

    assert find_native_artifacts(tmp_path) == [
        'libminimal_cuda_cnn.dylib',
        'libminimal_cuda_cnn.so',
        'minimal_cuda_cnn_handmade.dll',
    ]


def test_healthcheck_reports_windows_native_artifacts(monkeypatch, tmp_path: Path):
    import minicnn.framework.health as health

    cpp_root = tmp_path / 'cpp'
    cpp_root.mkdir()
    (cpp_root / 'minimal_cuda_cnn_handmade.dll').write_text('', encoding='utf-8')

    project_root = tmp_path / 'project'
    project_root.mkdir()
    data_root = tmp_path / 'data'

    monkeypatch.setattr(health, 'CPP_ROOT', cpp_root)
    monkeypatch.setattr(health, 'PROJECT_ROOT', project_root)
    monkeypatch.setattr(health, 'DATA_ROOT', data_root)
    monkeypatch.setattr(health, 'describe_registries', lambda: {'layers': ['conv2d']})
    monkeypatch.setattr(health, 'CUDA_LEGACY_SUPPORTED', {'conv2d': True})

    payload = health.healthcheck()
    artifact_check = next(check for check in payload['checks'] if check['name'] == 'native_cuda_artifacts')

    assert payload['diagnostic_kind'] == 'environment_diagnostic_summary'
    assert payload['native_artifacts'] == ['minimal_cuda_cnn_handmade.dll']
    assert payload['shared_objects'] == ['minimal_cuda_cnn_handmade.dll']
    assert artifact_check['ok'] is True
    assert artifact_check['details']['native_artifacts'] == ['minimal_cuda_cnn_handmade.dll']
    assert artifact_check['details']['shared_objects'] == ['minimal_cuda_cnn_handmade.dll']


def test_build_diagnostic_payload_keeps_ok_when_only_optional_checks_fail():
    from minicnn.framework.health import _check, build_diagnostic_payload

    payload = build_diagnostic_payload(
        checks=[
            _check('required_ok', True, required=True),
            _check('optional_missing', False, required=False),
        ]
    )

    assert payload['status'] == 'ok'
    assert payload['summary_status'] == 'ok'
    assert payload['warnings'] == ['optional_missing']
    assert payload['errors'] == []
