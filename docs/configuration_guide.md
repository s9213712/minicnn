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

## Dataset augmentation

CIFAR-10 torch flex configs may enable standard lightweight augmentation from
YAML:

```yaml
dataset:
  random_crop_padding: 4
  horizontal_flip: true
```

The legacy CUDA trainer exposes the same knobs through its compiled
experiment config and environment overrides:

```bash
MINICNN_RANDOM_CROP_PADDING=4 minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

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

The torch flex builder can infer `in_channels`, `num_features`, and
`in_features` for common layers. It supports the sequential layer subset used
by `configs/alexnet_like.yaml`, plus torch-only residual blocks in
`configs/resnet_like.yaml`:

```yaml
model:
  layers:
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
      padding: 1
      bias: false
    - type: BatchNorm2d
    - type: ReLU
    - type: ResidualBlock
      channels: 64
      stride: 1
    - type: GlobalAvgPool2d
    - type: Flatten
    - type: Linear
      out_features: 10
```

`ResidualBlock`, `BatchNorm2d`, and `GlobalAvgPool2d` are currently torch flex
components. The `cuda_legacy` backend still accepts only the supported CNN
subset reported by `minicnn validate-dual-config`.

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
