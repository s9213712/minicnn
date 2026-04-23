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
