from __future__ import annotations

import json
from pathlib import Path


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
    run_dir = Path(out.strip().split('Artifacts written to:', 1)[1].strip())
    assert run_dir.exists()
    assert run_dir.parent == tmp_path
    assert (run_dir / 'summary.json').exists()
    assert (run_dir / 'config.yaml').exists()

    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    best_model_path = Path(summary['best_model_path'])
    assert best_model_path.exists()
