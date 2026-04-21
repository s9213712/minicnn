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
- [guide_autograd.md](guide_autograd.md)
- [custom_components.md](custom_components.md)

---

# 功能擴展說明（中文）

本文整理圍繞原始 MiniCNN 核心所擴展的功能，以及各 backend 路徑目前實際支援哪些功能。

重要區別：功能擴展主要發生在前端、torch/flex 路徑和 NumPy autograd 路徑。`cuda_legacy` 刻意維持狹窄。

## 高層次概覽

| 功能區域 | Torch/flex | Autograd | `cuda_legacy` |
|---|---|---|---|
| 更多 activation | ✓ | ✓ | 僅有限子集 |
| 更多 optimizer | ✓ | ✓ | 僅 `SGD`、`Adam` |
| Scheduler | ✓ | ✓ | 無共用 scheduler bridge |
| Label smoothing | ✓ | ✓ | ✗ |
| Data augmentation | ✓ | ✗ | ✗ |
| Layer preset / 自訂 factory | ✓ | ✗ | ✗ |
| 更豐富的正規化層 | ✓ | 部分 | validator 拒絕 |

## Activation

前端與 autograd 擴展新增：

- `LeakyReLU`
- `SiLU`
- 既有：`ReLU`、`Sigmoid`、`Tanh`

`cuda_legacy` 只接受已驗證的固定 pattern，且 activation 只允許 `ReLU` 或 `LeakyReLU`。

## Optimizer

Torch/flex 與 autograd 支援更廣的 optimizer：

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`

`cuda_legacy` 目前支援：
- `SGD`
- `Adam`

另外支援 `optimizer.grad_clip_global`，但這是 backend 特定擴展，不是前端通用對等信號。

## Scheduler

Torch/flex 與 autograd 支援：

- `StepLR`
- `CosineAnnealingLR`

`cuda_legacy` 目前不消費共用的 scheduler 區塊。

## Loss 與正規化

Torch/flex 支援：
- `CrossEntropyLoss`
- `MSELoss`
- `BCEWithLogitsLoss`
- cross entropy label smoothing

Autograd 支援相同三種 loss 以及 label smoothing。

`cuda_legacy` 支援 `CrossEntropyLoss` 與 `MSELoss`；拒絕 `BCEWithLogitsLoss`，不支援 label smoothing。

## 資料增強

Torch/flex 透過 `minicnn.flex.data.create_dataloaders(...)` 提供輕量增強：

- random crop
- horizontal flip

此增強層未與 autograd 或 `cuda_legacy` 共用。

## Block Preset 與自訂元件

Torch/flex 透過 flex builder 與 registry 支援 preset 或 dotted-path 擴展：

- `conv_relu`
- `conv_bn_relu`
- `conv_bn_silu`
- 自訂 dotted-path factory

這些是前端便利功能，不會自動對 `cuda_legacy` 合法。

## 正規化與更豐富的 Layer

Torch/flex 可使用：`BatchNorm2d`、`LayerNorm`、`GroupNorm`、`ResidualBlock`

Autograd 目前支援：`BatchNorm2d`、`ResidualBlock`

`cuda_legacy` validator 拒絕：`BatchNorm2d`、`LayerNorm`、`GroupNorm`、`ResidualBlock`

## 相關文件

- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)
- [guide_autograd.md](guide_autograd.md)
- [custom_components.md](custom_components.md)
