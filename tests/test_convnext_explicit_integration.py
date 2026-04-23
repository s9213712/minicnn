from __future__ import annotations


def test_train_flex_supports_convnext_explicit_smoke_template(tmp_path, capsys):
    from minicnn.cli import main

    rc = main([
        'train-flex',
        '--config',
        'templates/cifar10/convnext_explicit_smoke.yaml',
        f'project.artifacts_root={tmp_path}',
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert 'Artifacts written to:' in out
