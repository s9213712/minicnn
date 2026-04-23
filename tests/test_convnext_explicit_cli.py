from __future__ import annotations

import json


def test_show_model_supports_convnext_explicit_template(capsys):
    from minicnn.cli import main

    rc = main([
        'show-model',
        '--config',
        'templates/cifar10/convnext_explicit.yaml',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['command'] == 'show-model'
    assert payload['status'] == 'ok'
    assert any(layer.get('type') == 'DepthwiseConv2d' for layer in payload['layers'])
    assert any(layer.get('type') == 'LayerNorm2d' for layer in payload['layers'])
    assert any(layer.get('type') == 'PointwiseConv2d' for layer in payload['layers'])


def test_show_graph_supports_convnext_explicit_template(capsys):
    from minicnn.cli import main

    rc = main([
        'show-graph',
        '--config',
        'templates/cifar10/convnext_explicit.yaml',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['command'] == 'show-graph'
    assert payload['status'] == 'ok'
    ops = [node.get('op_type') for node in payload.get('graph', {}).get('nodes', [])]
    assert 'DepthwiseConv2d' in ops
    assert 'LayerNorm2d' in ops
    assert 'PointwiseConv2d' in ops


def test_validate_config_supports_convnext_explicit_template(capsys):
    from minicnn.cli import main

    rc = main([
        'validate-config',
        '--config',
        'templates/cifar10/convnext_explicit.yaml',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['command'] == 'validate-config'
    assert payload['status'] == 'ok'
    assert payload['backend'] == 'torch'
    assert payload['ok'] is True
