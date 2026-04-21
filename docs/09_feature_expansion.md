# Feature Expansion Surface

This document summarizes the broader feature work that was added around the
original MiniCNN core, and which backend paths can actually use those features
today.

The important distinction is that feature expansion happened mostly in the
frontend, the torch/flex path, and the NumPy autograd path. `cuda_legacy`
remains intentionally narrow.

## High-Level View

| Area | Torch/flex | Autograd | `cuda_legacy` |
|---|---|---|---|
| Broader activations | yes | yes | narrow subset only |
| More optimizers | yes | yes | `SGD`, `Adam` only |
| Schedulers | yes | yes | no shared scheduler bridge |
| Label smoothing | yes | yes | no |
| Data augmentation | yes | no | no shared augmentation bridge |
| Layer presets / custom factories | yes | no | no |
| Richer normalization layers | yes | partial | validator rejects them |

## Activations

The broader frontend and autograd work added:

- `LeakyReLU`
- `SiLU`
- existing `ReLU`, `Sigmoid`, `Tanh`

Example:

```yaml
model:
  layers:
    - type: Conv2d
      out_channels: 16
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: SiLU
```

`cuda_legacy` only accepts the validated fixed pattern and only allows
`ReLU` or `LeakyReLU` inside that pattern.

## Optimizers

Torch/flex and autograd support a wider optimizer surface:

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`

Example:

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.01
```

`cuda_legacy` currently supports:

- `SGD`
- `Adam`

It also supports `optimizer.grad_clip_global`, but that is a backend-specific
extension, not a broad frontend parity signal.

## Schedulers

Torch/flex and autograd support:

- `StepLR`
- `CosineAnnealingLR`

Examples:

```yaml
scheduler:
  enabled: true
  type: StepLR
  step_size: 10
  gamma: 0.5
```

```yaml
scheduler:
  enabled: true
  type: CosineAnnealingLR
  T_max: 30
  min_lr: 1.0e-5
```

`cuda_legacy` on this branch does not consume the shared scheduler section.

## Losses and Regularization

Torch/flex supports:

- `CrossEntropyLoss`
- `MSELoss`
- `BCEWithLogitsLoss`
- label smoothing for cross entropy

Autograd supports the same three losses and also supports label smoothing on the
cross-entropy path.

`cuda_legacy` currently supports:

- `CrossEntropyLoss`
- `MSELoss`

It rejects `BCEWithLogitsLoss`, and it does not support label smoothing.

## Data Augmentation

Torch/flex currently exposes lightweight augmentation through
`minicnn.flex.data.create_dataloaders(...)`:

- random crop
- horizontal flip

Example:

```yaml
augmentation:
  random_crop: true
  random_crop_padding: 4
  horizontal_flip: true
```

This augmentation layer is not shared with autograd or `cuda_legacy`.

## Block Presets and Custom Components

Torch/flex supports preset-style or dotted-path expansion through the flex
builder and registry. Examples in the repo include:

- `conv_relu`
- `conv_bn_relu`
- `conv_bn_silu`
- custom dotted-path factories

Example:

```yaml
model:
  layers:
    - type: conv_bn_relu
      out_channels: 32
      kernel_size: 3
      padding: 1
```

These are frontend conveniences. They do not automatically become legal for
`cuda_legacy`.

## Normalization and Richer Layers

Torch/flex can use:

- `BatchNorm2d`
- `LayerNorm`
- `GroupNorm`
- `ResidualBlock`

Autograd currently supports:

- `BatchNorm2d`
- `ResidualBlock`

`cuda_legacy` validation rejects:

- `BatchNorm2d`
- `LayerNorm`
- `GroupNorm`
- `ResidualBlock`

`cuda_native` work exists on this branch, but it is still experimental and is
not part of the stable training backend surface.

## Useful Configs

- `configs/flex_broad.yaml`
- `configs/autograd_enhanced.yaml`
- `configs/cuda_legacy_strict.yaml`

## Related Docs

- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)
- [08_autograd.md](08_autograd.md)
- [custom_components.md](custom_components.md)
