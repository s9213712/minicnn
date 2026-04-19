# MiniCNN

[English README](README.md)

MiniCNN 是一個雙後端的小型深度學習框架。它讓同一份模型 config 可以切換到兩條路徑：

- **`engine.backend: torch`**：用 PyTorch 快速實驗，支援彈性 layer 與自訂元件。
- **`engine.backend: cuda_legacy`**：使用本專案內的手寫 CUDA CNN 路徑。

目標很直接：使用者只要改 config 裡的一個選項，就能切換 backend，同時保留同一份 layer 定義、optimizer、loss 與 training 參數。

這個 repository 是整理後的 MiniCNN 主線。舊的探索版本已移除，讓後續開發集中在單一穩定目標上。

## 專案內容

- `cpp/`：手寫 CUDA/C++ backend
- `src/minicnn/flex/`：config-driven PyTorch model builder
- `src/minicnn/unified/`：dual-backend config compiler 與 trainer
- GitHub-ready 專案結構：CI、docs、examples、tests

## 快速開始

```bash
git clone https://github.com/s9213712/minicnn.git
cd minicnn
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[torch,dev]
pytest
```

## 編譯手寫 CUDA shared library

```bash
minicnn build --legacy-make --check
```

也可以走 CMake path：

```bash
minicnn build --check
```

## 使用同一份 config 切換 backend

### 1. PyTorch backend

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
```

### 2. 手寫 CUDA backend

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

上面兩個指令唯一需要切換的是 `engine.backend`。

## 為什麼有兩個 backend

本專案刻意保留兩種 workflow：

- **Torch backend**：支援較廣的 layer、快速實驗、自訂 dotted-path component。
- **CUDA backend**：保留手寫 CUDA CNN 路徑，方便低階控制與 backend ownership。

## Shared Config Contract

同一份 config 包含：

- `dataset`
- `model.layers`
- `train`
- `loss`
- `optimizer`
- `scheduler`
- `engine.backend`

範例：

```yaml
engine:
  backend: torch

model:
  layers:
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Flatten
    - type: Linear
      out_features: 10
```

## CUDA Backend 支援範圍

手寫 CUDA 路徑目前支援 `src/minicnn/unified/cuda_legacy.py` 可以編譯的 subset：

- dataset：`cifar10`
- layers：`Conv2d -> activation -> Conv2d -> activation -> MaxPool2d -> Conv2d -> activation -> Conv2d -> activation -> MaxPool2d -> Flatten -> Linear`
- activations：`ReLU` 或 `LeakyReLU`，並使用單一 shared negative slope
- optimizer：`SGD`
- loss：`CrossEntropyLoss`
- input shape：`[3, 32, 32]`
- classes：`10`

如果 config 超出這個 subset，`validate-dual-config` 會說明原因。

## 常用指令

```bash
minicnn info
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn dual-config-template
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

## 像 PyTorch 一樣使用自訂元件

Torch backend 允許在 config 中直接指定 dotted-path class。

範例：

```yaml
model:
  layers:
    - type: Flatten
    - type: Linear
      out_features: 32
    - type: examples.custom_block.CustomHead
      in_features: 32
      out_features: 10
```

執行：

```bash
minicnn train-dual --config configs/dual_backend_torch_custom.yaml
```

## 專案結構

```text
configs/
cpp/
docs/
examples/
src/minicnn/
  config/
  core/
  data/
  engine/
  flex/
  framework/
  nn/
  optim/
  runtime/
  schedulers/
  training/
  unified/
tests/
```

## 開發

```bash
python -m pip install -e .[torch,dev]
pytest
python -m compileall -q src
```

## Feature 隔離研發流程

穩定程式碼放在 `src/minicnn/`，並且 `main` 必須維持可執行。新的或高風險功能先開 Git branch，並放在 `features/` 下隔離研發。

```bash
git checkout -b feature/native-cuda-class-backend
mkdir -p features/native-cuda-class-backend
```

`features/<name>/` 用來放 prototype、notes、探索性測試。正式 CLI 預設不應 import `features/`。功能穩定後，再把支援的實作移到 `src/minicnn/`，測試移到 `tests/`，更新 README/docs，並在合併前跑完整測試。

大型實驗建議使用 worktree，讓穩定版 checkout 保持可用：

```bash
git worktree add ../minicnn-feature-native -b feature/native-backend
```

## 備註

- **Torch path 是最彈性的路徑**。
- **CUDA path 是手寫 backend 路徑**，目前會先驗證 config 是否在支援範圍內。
- 這樣可以維持統一 config 介面，同時誠實標示不同 backend 的能力邊界。

## 能力邊界說明

這個 package 提供兩個 backend 共用的 config 介面。Torch path 完整彈性較高。手寫 CUDA path 是真的 CUDA backend，但目前只支援上面描述的 CNN subset，並會將該 subset 編譯到 legacy CUDA trainer。
