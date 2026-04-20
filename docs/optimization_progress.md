# MiniCNN Optimization Progress

Last updated: 2026-04-20

This file records the current optimization state for the MiniCNN follow-up work
after comparing the project with `llm.c`, `tiny-cuda-nn`, and
`neural-network-cuda`.

## Current PR

| Field | Value |
|---|---|
| PR | `https://github.com/s9213712/minicnn/pull/1` |
| Branch | `codex/comparison-completion-report` |
| Latest uploaded commit | `66082d3 Add CUDA legacy global grad clipping` |
| CI status | GitHub Actions passed on Python 3.10, 3.11, and 3.12 |

## Completed

| Area | Status |
|---|---|
| Comparison report | Added backend-scoped completion report and linked it from README |
| Follow-up tracking | Follow-up items tracked inline (comparison_followup_todo.md removed) |
| Backend support matrix | Added `docs/backend_capabilities.md` |
| Benchmark reporting | Added `docs/benchmark_report_template.md` |
| `minicnn compare` UX | Added benchmark-ready fields and fixed `--backends ... key=value` parsing |
| MNIST ctypes examples | Moved train scripts from `docs/` to `examples/mnist_ctypes/` |
| Autograd losses | Added CPU/NumPy `MSELoss` and `BCEWithLogitsLoss` |
| Autograd train config | `train-autograd` can select CrossEntropy, MSE, or BCEWithLogits |
| PyTorch parity tests | Added parity tests for Linear, BatchNorm2d, MSE, and BCEWithLogits |
| `Tensor.__pow__` warnings | Removed zero-base negative-exponent RuntimeWarnings |
| CUDA legacy global grad clipping | Added `optimizer.grad_clip_global`, default disabled with `0.0` |
| CUDA legacy BatchNorm2d | Evaluated scope and documented integration requirements |

## CUDA Legacy Global Grad Clipping

Implemented as a low-risk CUDA legacy optimizer improvement.

- Config key: `optimizer.grad_clip_global`
- Default: `0.0`, disabled
- Validation accepts numeric values through unified config overrides
- The CUDA batch step now computes FC and Conv gradients before applying
  updates, derives one global norm scale, then applies the existing CUDA update
  kernels through adjusted gradient normalizers
- No native ABI was added for this step

Example:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy optimizer.grad_clip_global=2.0
```

## CUDA Legacy BatchNorm2d Status

BatchNorm2d is not blocked by theory, but it is not a one-kernel change. The
current CUDA legacy training graph has no BN state, BN workspace, BN checkpoint
fields, or Conv-BN-Activation call path.

Current status:

- Torch/flex backend: supported
- CPU/NumPy autograd backend: supported
- CUDA legacy backend: not supported yet

Current user-facing behavior:

```text
cuda_legacy does not yet support BatchNorm2d in the training graph; use engine.backend=torch or remove BatchNorm2d for cuda_legacy
```

The implementation plan is recorded in
`docs/cuda_batchnorm2d_evaluation.md`.

## Validation

Latest local validation before upload:

```text
PYTHONPATH=/home/s92137/NN/minicnn/src python3 -m pytest -q /home/s92137/NN/minicnn/tests
117 passed

python3 -m compileall -q /home/s92137/NN/minicnn/src /home/s92137/NN/minicnn/examples/mnist_ctypes
passed

git -C /home/s92137/NN/minicnn diff --check
passed
```

Latest GitHub Actions validation for commit `66082d3`:

```text
test (3.10): success
test (3.11): success
test (3.12): success
```

## Remaining Work

| Priority | Item | Notes |
|---:|---|---|
| 1 | CUDA Adam or AdamW | Needs optimizer state and update kernels |
| 2 | CUDA BatchNorm2d | Needs kernels, runtime state, workspace, checkpoints, validation, and parity tests |
| 3 | CUDA residual/skip add | Needs compiler/runtime representation first |
| 4 | CUDA FP16/AMP | Deferred until baseline CUDA path is stronger |
| 5 | Multi-GPU / NCCL / distributed | Deferred |
