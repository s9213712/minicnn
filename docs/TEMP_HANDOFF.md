# Temporary Handoff

Created because usage quota was reported near exhaustion.

## Current State

- Date: 2026-04-19
- Branch: `main`
- Remote: `origin/main`
- Latest pushed commit before this handoff: `f9b6430 record torch smoke test metrics`
- Working tree before creating this file had no tracked changes.
- Ignored local-only data:
  - `artifacts/.gitkeep`
  - `data/cifar-10-batches-py`

## Completed Work

- Native CUDA `.so` loading is lazy, so non-CUDA commands work without a built shared library.
- CUDA legacy training helpers were split into `cuda_ops.py` and `cuda_workspace.py`.
- Common legacy training parameters support environment overrides.
- Windows native build support and docs exist.
- Model outputs now go to `src/minicnn/training/models/`.
- README and docs include folder/file purpose tables and training entrypoint locations.
- Torch smoke metrics now include `epoch_time_s` and CIFAR-10 official `test_acc`.
- `features/backend-smoke-matrix/` is the example isolated feature folder.
- Completed comment files have been removed from `comments/`.

## Latest Verification

Commands run before this handoff:

```bash
PYTHONPATH=/home/s92137/NN/minicnn/src python3 -m pytest -q --rootdir=/home/s92137/NN/minicnn /home/s92137/NN/minicnn/tests
python3 -m compileall -q /home/s92137/NN/minicnn/src /home/s92137/NN/minicnn/features/backend-smoke-matrix/run_smoke_matrix.py
```

Results:

- `pytest`: `15 passed`
- `compileall`: passed

## Resume Notes

- Start with `git -C /home/s92137/NN/minicnn status --short --ignored --untracked-files=all`.
- If CUDA smoke testing is needed, rebuild generated `.so` files first:

```bash
PYTHONPATH=src python3 -m minicnn.cli build --legacy-make --variant both --check
python3 features/backend-smoke-matrix/run_smoke_matrix.py
```

- Generated files should stay out of git:
  - `cpp/libminimal_cuda_cnn*.so`
  - `src/minicnn/training/models/*.pt`
  - `src/minicnn/training/models/*.npz`
  - `__pycache__/`
  - `.pytest_cache/`

