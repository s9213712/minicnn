# Dual Backend Guide

This guide explains how the shared dual-backend config works.

The stable shared toggles are:

- `engine.backend: torch`
- `engine.backend: cuda_legacy`
- `engine.backend: cuda_native` *(experimental)*

## Same Frontend, Different Backend Boundary

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
minicnn train-native --config configs/dual_backend_cnn.yaml   # cuda_native shortcut
```

Both commands read the same top-level config shape. Torch can use far more of that frontend surface. `cuda_legacy` and `cuda_native` only accept narrow validated subsets.

## Validate Before Running

For `cuda_legacy`:

```bash
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
```

For `cuda_native`:

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

Do not assume any backend accepts the same full surface as torch.

## Inspect The Mapping

```bash
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

Shows how the shared config is compiled into the legacy experiment config.

## What The Shared Config Actually Uses

Architecture is described through `model.layers[]`.

- **Torch**: consumes the list directly through the flex builder
- **cuda_legacy**: validates the list against one fixed pattern, then compiles it into the stage-oriented experiment config
- **cuda_native**: validates against supported op types, then builds a sequential NativeGraph

The same YAML key exists on all sides, but the accepted semantic surface is different.

## cuda_legacy Contract

Accepts:

- dataset type: `cifar10`, input shape `[3, 32, 32]`
- exactly: `Conv2d → activation → Conv2d → activation → MaxPool2d → Conv2d → activation → Conv2d → activation → MaxPool2d → Flatten → Linear`
- activations: `ReLU` or `LeakyReLU`

## cuda_native Contract

Accepts:

- dataset type: `cifar10`, `mnist`, or `random`
- any sequential graph with supported ops: `BatchNorm2d` (forward/backward prototype), `Conv2d`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `SiLU`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `Linear`
- loss type: `CrossEntropyLoss` or `MSELoss`
- optimizer: `SGD` with optional momentum and global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- requires: `train.amp=false`, `train.grad_accum_steps=1`
- rejects at validation: `GroupNorm`, `LayerNorm`, `ResidualBlock`

## Variant Selection (cuda_legacy)

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

In-process variant switching resets the cached native handle so repeated scripted comparisons still load the requested `.so`.

## Changing Architecture

### Torch

Most torch-side architecture edits are YAML-only. PyTorch autograd owns backward for those components.

### cuda_legacy

Safe YAML-only edits: channel widths, `ReLU` vs `LeakyReLU`, classifier output width.

Not YAML-only: new layer types, different pool structure, branching topology. Those require backend work across `cuda_legacy.py`, `training/`, `core/`, and `cpp/src/`.

### cuda_native

Add a new op: implement kernel in `kernels.py`, add backward in `backward.py`, register in `capabilities.py` and `validators.py`, add shape inference in `shapes.py`.

## Change-Impact Table

| Change | Torch | cuda_legacy | cuda_native |
|---|---|---|---|
| Change Conv2d / Linear widths inside supported shapes | YAML-only | YAML-only if inside validator boundary | YAML-only |
| Add arbitrary layers | Usually YAML-only | Not supported without backend work | Only if op is in supported set |
| Add new native training op | No torch change needed | Update validator, mapping, workspace, native kernels | Update kernels, backward, shapes, validators |
| Change optimizer semantics | Torch optimizer/config changes | Native optimizer and CUDA batch path changes | Update `training.py` |

---

# Dual Backend 使用指南（中文）

本指南說明 shared dual-backend config 的使用方式。

穩定的 backend 切換項目：

- `engine.backend: torch`
- `engine.backend: cuda_legacy`
- `engine.backend: cuda_native` *(實驗中)*

## 相同前端，不同 Backend 邊界

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
minicnn train-native --config configs/dual_backend_cnn.yaml   # cuda_native 快捷方式
```

所有指令都讀同一份 config。Torch 能使用最多的前端功能；`cuda_legacy` 和 `cuda_native` 只接受各自驗證過的子集。

## 執行前先驗證

`cuda_legacy` 驗證：

```bash
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
```

`cuda_native` 驗證：

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

不要假設任何 backend 接受和 torch 一樣廣的前端功能。

## 查看 Mapping

```bash
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

顯示 shared config 如何被編譯成 legacy experiment config。

## Shared Config 實際使用方式

架構透過 `model.layers[]` 描述：

- **Torch**：透過 flex builder 直接使用
- **cuda_legacy**：對照固定 pattern 驗證後，編譯成 stage-oriented experiment config
- **cuda_native**：對照支援 op 列表驗證後，建立 sequential NativeGraph

同一個 YAML key 在各個 backend 的接受範圍不同。

## cuda_legacy 支援邊界

接受：
- 資料集：`cifar10`，input shape `[3, 32, 32]`
- 固定 pattern：`Conv2d → activation → Conv2d → activation → MaxPool2d → Conv2d → activation → Conv2d → activation → MaxPool2d → Flatten → Linear`
- 激活：`ReLU` 或 `LeakyReLU`

## cuda_native 支援邊界

接受：
- 資料集：`cifar10`、`mnist`、`random`
- 任何 sequential graph，op 限於：`BatchNorm2d`（forward/backward prototype）、`Conv2d`、`ReLU`、`LeakyReLU`、`Sigmoid`、`Tanh`、`SiLU`、`MaxPool2d`、`AvgPool2d`、`Flatten`、`Linear`
- loss：`CrossEntropyLoss` 或 `MSELoss`
- optimizer：支援 `SGD`，可選 momentum 與 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- 仍要求：`train.amp=false`、`train.grad_accum_steps=1`
- 驗證時拒絕：`GroupNorm`、`LayerNorm`、`ResidualBlock`

## Variant 選擇（cuda_legacy）

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

切換 variant 會重置 cached native handle，確保多次比較時正確載入指定的 `.so`。

## 修改架構的影響

### Torch

大多數架構修改只需改 YAML，backward 由 PyTorch autograd 負責。

### cuda_legacy

安全的 YAML-only 修改：channel 寬度、`ReLU` vs `LeakyReLU`、分類器輸出寬度。

非 YAML-only：新層型別、不同 pool 結構、branching topology，需要修改 `cuda_legacy.py`、`training/`、`core/`、`cpp/src/`。

### cuda_native

新增 op：在 `kernels.py` 實作 kernel，在 `backward.py` 加 backward，在 `capabilities.py` 和 `validators.py` 更新支援清單，在 `shapes.py` 加 shape inference。

## 修改影響一覽表

| 修改項目 | Torch | cuda_legacy | cuda_native |
|---|---|---|---|
| 改 Conv2d / Linear 寬度 | YAML-only | 在 validator 邊界內 YAML-only | YAML-only |
| 新增任意層 | 通常 YAML-only | 需要 backend 工作 | 僅限已支援 op |
| 新增 native 訓練 op | 不需 torch 修改 | 需更新 validator、mapping、workspace、native kernel | 需更新 kernels、backward、shapes、validators |
| 修改 optimizer 語意 | 改 torch optimizer/config | 改 native optimizer 和 CUDA batch 路徑 | 改 `training.py` |
