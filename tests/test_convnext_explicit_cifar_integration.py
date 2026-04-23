from __future__ import annotations

from pathlib import Path

import pytest


def test_train_flex_supports_convnext_explicit_cifar_template_when_data_is_prepared(tmp_path, capsys):
    from minicnn.cli import main
    from minicnn.data.cifar10 import cifar10_ready
    from minicnn.paths import DATA_ROOT

    if not cifar10_ready(DATA_ROOT):
        pytest.skip('requires prepared CIFAR-10 data under project data root')

    rc = main([
        'train-flex',
        '--config',
        'templates/cifar10/convnext_explicit.yaml',
        'train.epochs=1',
        'train.batch_size=16',
        'train.device=cpu',
        'dataset.num_samples=64',
        'dataset.val_samples=16',
        f'project.artifacts_root={tmp_path}',
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert 'Artifacts written to:' in out
    run_dir = Path(out.strip().split('Artifacts written to:', 1)[1].strip())
    assert run_dir.exists()
    assert (run_dir / 'summary.json').exists()
