# MiniCNN Benchmark Report Template

Use this template when comparing backend performance. Keep benchmark reports
small, repeatable, and tied to exact commands.

## Environment

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Host | |
| GPU | |
| Driver | |
| CUDA toolkit | |
| Python | |
| PyTorch | |
| MiniCNN commit | |

## Commands

```bash
minicnn build --legacy-make --variant both --check
```

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=torch train.epochs=1 dataset.num_samples=1024 dataset.val_samples=256
```

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas \
  train.epochs=1 dataset.num_samples=1024 dataset.val_samples=256
```

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade \
  train.epochs=1 dataset.num_samples=1024 dataset.val_samples=256
```

## Results

| Backend | Variant | Train samples | Val samples | Batch size | Epoch time | Samples/sec | Peak GPU memory | Train acc | Val acc | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| torch | PyTorch CUDA | | | | | | | | | |
| cuda_legacy | cublas | | | | | | | | | |
| cuda_legacy | handmade | | | | | | | | | |

## Correctness Checks

- [ ] `python -m compileall -q src`
- [ ] `pytest -q`
- [ ] `minicnn build --legacy-make --variant both --check`
- [ ] Relevant parity tests passed
- [ ] Backend smoke matrix passed, if used

## Interpretation

Summarize what changed and whether the result is expected. Do not compare runs
that used different sample counts, batch sizes, initial seeds, augmentation, or
native library variants without calling that out explicitly.

## Follow-up

- [ ] Unexpected slowdown investigated
- [ ] Any accuracy regression explained
- [ ] CUDA memory growth checked for leaks
