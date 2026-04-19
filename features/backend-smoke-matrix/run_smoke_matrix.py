#!/usr/bin/env python3
"""Prototype backend smoke matrix runner.

This script is intentionally outside src/minicnn. It exercises supported CLI
entrypoints without becoming part of the production package.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path('/tmp/minicnn_feature_backend_smoke_matrix')

COMMON_OVERRIDES = [
    '--config', 'configs/dual_backend_cnn.yaml',
    'train.epochs=1',
    'dataset.num_samples=128',
    'dataset.val_samples=32',
    'train.batch_size=32',
    f'project.artifacts_root={ARTIFACT_ROOT}',
]

JOBS = [
    ('torch-cuda', ['engine.backend=torch', 'train.device=cuda', 'project.run_name=feature-torch-cuda']),
    ('cuda-cublas', ['engine.backend=cuda_legacy', 'runtime.cuda_variant=cublas', 'project.run_name=feature-cuda-cublas']),
    ('cuda-handmade', ['engine.backend=cuda_legacy', 'runtime.cuda_variant=handmade', 'project.run_name=feature-cuda-handmade']),
]


def torch_cuda_available() -> bool:
    code = 'import torch; print(int(torch.cuda.is_available()))'
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, check=False)
    return result.returncode == 0 and result.stdout.strip().endswith('1')


def run_job(name: str, overrides: list[str]) -> int:
    cmd = [sys.executable, '-u', '-m', 'minicnn.cli', 'train-dual', *COMMON_OVERRIDES, *overrides]
    env = {**os.environ, 'PYTHONPATH': str(REPO_ROOT / 'src')}
    print(f'\n=== {name} ===', flush=True)
    print(' '.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return result.returncode


def main() -> int:
    if not torch_cuda_available():
        print('CUDA is not available to this Python process; skipping GPU smoke matrix.')
        return 0

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, overrides in JOBS:
        failures += int(run_job(name, overrides) != 0)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
