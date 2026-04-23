from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'


def test_train_flex_from_outside_repo_root_uses_project_root_relative_artifacts(tmp_path):
    config_path = REPO_ROOT / 'templates' / 'cifar10' / 'convnext_explicit_smoke.yaml'
    artifact_root = REPO_ROOT / 'path-policy-artifacts'
    if artifact_root.exists():
        for path in sorted(artifact_root.rglob('*'), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        artifact_root.rmdir()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'minicnn.cli',
            'train-flex',
            '--config',
            str(config_path),
            'project.artifacts_root=path-policy-artifacts',
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert 'Artifacts written to:' in proc.stdout
    run_dir = Path(proc.stdout.strip().split('Artifacts written to:', 1)[1].strip())
    assert run_dir.exists()
    assert artifact_root in run_dir.parents
