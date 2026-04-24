from __future__ import annotations

import json


def test_run_smoke_checks_exposes_dependency_flags():
    from minicnn._cli_readonly import run_smoke_checks

    payload = run_smoke_checks()

    assert 'torch_available' in payload
    assert 'cuda_available' in payload
    assert 'native_available' in payload
    assert 'flex_registry_ready' in payload
    assert 'warnings' in payload
    assert 'errors' in payload


def test_run_smoke_checks_warns_when_torch_is_missing(monkeypatch):
    import minicnn._cli_readonly as cli_readonly

    monkeypatch.setattr(
        cli_readonly,
        'import_torch_with_details',
        lambda: (None, ModuleNotFoundError("No module named 'torch'", name='torch')),
    )

    payload = cli_readonly.run_smoke_checks()
    torch_check = next(check for check in payload['checks'] if check['name'] == 'torch_dependency')

    assert payload['torch_available'] is False
    assert payload['flex_registry_ready'] is False
    assert torch_check['required'] is False
    assert torch_check['severity'] == 'warning'
    assert 'pip install -e .[torch]' in torch_check['suggested_fix']


def test_cli_smoke_json_contract_exposes_registry_aliases_and_flags(capsys):
    from minicnn.cli import main

    rc = main(['smoke'])
    payload = json.loads(capsys.readouterr().out)

    assert rc in {0, 2}
    assert payload['command'] == 'smoke'
    assert 'flex_registry_ready' in payload
    assert 'dependency_status' in payload
    assert 'checks' in payload

    registry_check = next(check for check in payload['checks'] if check['name'] == 'flex_registry_surface')
    layers = registry_check['details']['registries']['layers']

    assert 'DepthwiseConv2d' in layers
    assert 'PointwiseConv2d' in layers
    assert 'LayerNorm2d' in layers
    assert 'convnext_block' in layers
    assert 'depthwise_conv2d' in layers
    assert 'pointwise_conv2d' in layers
    assert 'layernorm2d' in layers


def test_run_smoke_checks_reports_optional_cuda_legacy_artifact_precisely():
    from minicnn._cli_readonly import run_smoke_checks

    payload = run_smoke_checks()
    native_check = next(
        check for check in payload['checks']
        if check['name'] == 'optional_cuda_legacy_native_artifacts'
    )

    assert native_check['required'] is False
    assert native_check['details']['component'] == 'optional cuda_legacy native artifact'
    assert 'cuda_native_validation' in native_check['details']['available_surfaces']
    assert 'cuda_legacy_validation' in native_check['details']['available_surfaces']
