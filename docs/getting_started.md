# Getting Started

## Recommended path

For most users, start with the configuration-driven path:

```bash
python -m pip install -e .[torch]
minicnn healthcheck
minicnn list-flex-components
minicnn train-flex --config configs/flex_cnn.yaml
```

## Common workflow

1. Copy a config file from `configs/`.
2. Edit the model, optimizer, scheduler, and dataset sections.
3. Run `minicnn train-flex --config <your-config>.yaml`.
4. Inspect the run artifacts.

## Suggested first configs

- `configs/flex_cnn.yaml`: CNN-style image pipeline
- `configs/flex_mlp.yaml`: MLP baseline
- `configs/flex_custom.yaml`: custom dotted-path components
