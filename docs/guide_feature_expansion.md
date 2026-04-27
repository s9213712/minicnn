# Feature Expansion Surface

This document summarizes the broader feature work that was added around the
original MiniCNN core, and which backend paths can actually use those features
today.

The important distinction is that feature expansion happened mostly in the
frontend, the torch/flex path, and the NumPy autograd path. `cuda_legacy`
remains intentionally narrow.

`cuda_native` is now a beta public training backend with its own explicit
support matrix. This page stays frontend-oriented; when you need the exact
native subset, use [cuda_native.md](cuda_native.md) instead of inferring
support from the older `cuda_legacy` boundary.

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

On the autograd path, `optimizer.grad_clip` now means global-norm clipping
across all parameters, and `AdamW` uses decoupled weight decay ordering.

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
- `ReduceLROnPlateau`

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

## Reproducibility And Checkpoint Behavior

The broader training surface now also includes more explicit runtime semantics:

- `train.init_seed` controls model construction
- `train.train_seed` controls runtime stochasticity such as dropout and
  shuffled autograd batch order
- `minicnn.nn.set_global_seed(...)` gives the same seed control when you build
  modules directly from Python examples
- `Module.state_dict()` returns snapshot copies and includes registered buffers,
  so `BatchNorm2d` running stats survive checkpoint save/load

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

`cuda_native` is now a beta training backend with its own documented support
surface. Use [cuda_native.md](cuda_native.md) for the exact native subset
instead of treating this frontend-oriented summary as the native contract.

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

`cuda_native` 現在已是 beta 級的公開訓練 backend，並有獨立的支援矩陣。
本文仍以 frontend / feature expansion 的視角整理；若你要看精確的 native
subset，請直接看 [cuda_native.md](cuda_native.md)，不要從 `cuda_legacy`
的窄邊界反推。

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

在 autograd 路徑上，`optimizer.grad_clip` 現在採 global-norm clipping 語義；
`AdamW` 也已對齊 decoupled weight decay 的更新順序。

`cuda_legacy` 目前支援：
- `SGD`
- `Adam`

另外支援 `optimizer.grad_clip_global`，但這是 backend 特定擴展，不是前端通用對等信號。

## Scheduler

Torch/flex 與 autograd 支援：

- `StepLR`
- `CosineAnnealingLR`
- `ReduceLROnPlateau`

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

## 可重現性與 Checkpoint 行為

目前較廣的訓練 surface 也已補上較明確的 runtime 語義：

- `train.init_seed` 控制模型建構
- `train.train_seed` 控制 dropout 與 autograd batch shuffle 等訓練期隨機性
- 直接寫 Python 範例時，可用 `minicnn.nn.set_global_seed(...)` 套用相同 seed 控制
- `Module.state_dict()` 會回傳 snapshot copy，且包含 registered buffer，
  所以 `BatchNorm2d` 的 running stats 能正確跨 checkpoint 保留

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

`cuda_native` 現在已是 beta 級訓練 backend，且有獨立文件定義支援邊界。
若你要確認 native subset，請看 [cuda_native.md](cuda_native.md)，不要把這份
frontend-oriented 摘要當成 native contract。

## 相關文件

- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)
- [guide_autograd.md](guide_autograd.md)
- [custom_components.md](custom_components.md)
