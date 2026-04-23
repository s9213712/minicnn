# ConvNeXt Explicit Smoke / Test Index

This page is the compact index for MiniCNN's explicit ConvNeXt primitive path.

Use it when you want the smallest reliable workflow for:

- verifying the explicit primitive template surface
- checking CLI introspection output
- running a minimal training smoke path
- locating the regression tests that protect this path

## Templates

- `templates/cifar10/convnext_explicit.yaml`
  - full explicit primitive template
  - intended for direct editing and model experimentation
  - dataset-dependent CIFAR path, not the hermetic smoke path
- `templates/cifar10/convnext_explicit_smoke.yaml`
  - smallest built-in training smoke config
  - uses `dataset.type=random`, so it is hermetic
  - fixed to `num_samples=64`, `val_samples=16`, `epochs=1`, `batch_size=16`, `device=cpu`

## CLI checks

Use these when you want to inspect the explicit primitive path without starting a long run:

```bash
minicnn show-model --config templates/cifar10/convnext_explicit.yaml
minicnn show-graph --config templates/cifar10/convnext_explicit.yaml
minicnn validate-config --config templates/cifar10/convnext_explicit.yaml
```

## Minimal training smoke

```bash
minicnn train-flex --config templates/cifar10/convnext_explicit_smoke.yaml
```

This should complete one small CPU epoch and write an artifact directory under
`artifacts/`.

## Dataset-dependent CIFAR integration

Use this only when prepared CIFAR-10 data is already present under the project
data root:

```bash
minicnn train-flex --config templates/cifar10/convnext_explicit.yaml \
  train.epochs=1 train.batch_size=16 train.device=cpu \
  dataset.num_samples=64 dataset.val_samples=16
```

Path policy note:

- relative `dataset.data_root` and `project.artifacts_root` values are resolved
  against `PROJECT_ROOT`
- this stays stable even when `minicnn` is launched from outside the repo root

## Tests

- `tests/test_templates_validity.py`
  - template parsing, dataset split validity, explicit template forward path
- `tests/test_convnext_components.py`
  - explicit primitive registration and module-level forward coverage
- `tests/test_convnext_explicit_cli.py`
  - `show-model`, `show-graph`, `validate-config`
- `tests/test_convnext_explicit_smoke.py`
  - hermetic CLI `train-flex` smoke
- `tests/test_convnext_explicit_cifar_integration.py`
  - explicit CIFAR integration, only when prepared data exists

## Scope boundary

This path is intentionally limited to `torch/flex`.

It does not claim:

- `cuda_legacy` support
- `cuda_native` support
- full repo-wide ConvNeXt support
- silent fallback from unsupported backends
