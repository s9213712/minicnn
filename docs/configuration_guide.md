# Configuration Guide

MiniCNN V7 is designed so that users edit configuration instead of framework internals.

## Main sections

Typical top-level sections are:
- `dataset`
- `model`
- `loss`
- `optimizer`
- `scheduler`
- `train`

## Model section

```yaml
model:
  input_shape: [3, 32, 32]
  layers:
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
      padding: 1
    - type: ReLU
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Flatten
    - type: Linear
      out_features: 10
```

## Optimizer section

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.0001
```

## Scheduler section

```yaml
scheduler:
  type: CosineAnnealingLR
  T_max: 20
```

## CLI overrides

```bash
minicnn train-flex --config configs/flex_cnn.yaml train.epochs=3 optimizer.lr=0.0005
```

## Design rule

If a user needs to change architecture or common hyperparameters, they should usually be able to do it from YAML.
