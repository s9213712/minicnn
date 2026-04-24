from __future__ import annotations


def test_build_model_supports_named_convnext_tiny_entry():
    import torch

    from minicnn.flex.builder import build_model

    model = build_model({'name': 'convnext_tiny'}, input_shape=(3, 32, 32))
    y = model(torch.randn(2, 3, 32, 32))

    assert y.shape == (2, 10)


def test_validate_config_accepts_named_convnext_tiny_template(capsys):
    from minicnn.cli import main

    rc = main([
        'validate-config',
        '--config',
        'templates/cifar10/convnext_tiny.yaml',
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert '"ok": true' in out


def test_named_model_resolution_ignores_placeholder_layers():
    from minicnn.model_spec import resolve_model_config

    resolved = resolve_model_config({
        'name': 'convnext_tiny',
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ],
        'stem_channels': 16,
        'stage2_channels': 32,
    })

    layer_types = [layer['type'] for layer in resolved['layers']]

    assert layer_types[:3] == ['Conv2d', 'ConvNeXtBlock', 'ConvNeXtBlock']
    assert layer_types[-3:] == ['GlobalAvgPool2d', 'Flatten', 'Linear']
