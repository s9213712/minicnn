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
- `templates/cifar10/convnext_explicit_smoke.yaml`
  - smallest built-in training smoke config
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

## Tests

- `tests/test_templates_validity.py`
  - template parsing, dataset split validity, explicit template forward path
- `tests/test_convnext_components.py`
  - explicit primitive registration and module-level forward coverage
- `tests/test_convnext_explicit_cli.py`
  - `show-model`, `show-graph`, `validate-config`
- `tests/test_convnext_explicit_integration.py`
  - CLI `train-flex` integration smoke

## Scope boundary

This path is intentionally limited to `torch/flex`.

It does not claim:

- `cuda_legacy` support
- `cuda_native` support
- full repo-wide ConvNeXt support
- silent fallback from unsupported backends
