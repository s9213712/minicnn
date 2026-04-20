# Feature: backend-smoke-matrix

## Goal

Prototype a repeatable smoke benchmark that compares the same CIFAR-10 config across:

- `torch` on CUDA
- `cuda_legacy` with `runtime.cuda_variant=cublas`
- `cuda_legacy` with `runtime.cuda_variant=handmade`

This feature is intentionally isolated. It is a development aid, not a supported production CLI command yet.

## Isolation Boundary

Experimental files live under:

```text
features/backend-smoke-matrix/
```

Production code must not import this folder. If the prototype becomes useful, promote only the supported pieces into:

```text
src/minicnn/
tests/
docs/
```

## Prototype Command

From the repository root:

```bash
python3 features/backend-smoke-matrix/run_smoke_matrix.py
```

The script runs tiny one-epoch smoke jobs and writes artifacts under:

```text
/tmp/minicnn_feature_backend_smoke_matrix
```

Best model files still go to:

```text
src/minicnn/training/models/
```

## Expected Use

Use this after changing CUDA kernels, ctypes bindings, trainer scheduling, or backend config mapping. The goal is to catch obvious backend regressions quickly before running longer training.

The supported trainer code lives under `src/minicnn/training/`: CUDA
orchestration in `train_cuda.py`, CUDA batch device work in `cuda_batch.py`,
Torch baseline orchestration in `train_torch_baseline.py`, and shared loop
state in `loop.py`.

## Promotion Checklist

- The command is useful across multiple changes.
- Runtime options are stable.
- Results are written as structured JSON or Markdown.
- GPU-required cases are skipped cleanly when CUDA is unavailable.
- Tests are added under `tests/`.
- README/docs are updated.
