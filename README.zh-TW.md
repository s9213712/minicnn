# MiniCNN

[English README](README.md)

![status](https://img.shields.io/badge/status-experimental-orange)
![frontend](https://img.shields.io/badge/frontend-YAML%20%2B%20CLI-blue)
![native](https://img.shields.io/badge/native-CUDA-green)

MiniCNN 是一個以組態驅動的深度學習專案，用來研究「同一個前端介面」與
「不同 backend 能力邊界」之間的落差。

目前這個 repo 實際提供四條可用路徑：

- `torch`：透過 `train-flex` / `train-dual` 做較廣泛的模型實驗
- `cuda_legacy`：透過 `train-dual` 使用手寫 CUDA 的 CIFAR-10 訓練路徑
- `autograd`：透過 `train-autograd` 使用純 NumPy 的教學型 autograd stack
- `cuda_native`：透過 `train-native` 使用實驗性 graph-based backend（非正式）

## 為什麼有這個專案

大多數框架會刻意把 kernel orchestration、記憶體處理、backend 邊界包在
平滑的 API 後面。

MiniCNN 的價值在於把這些邊界攤開來看：

- 同一份前端設定介面如何映射到不同 backend 現實
- 狹窄 native backend 何時該嚴格驗證，而不是假裝功能對等
- 不依賴 torch internals 的小型 autograd stack 會怎麼運作
- 未來 graph-based native backend 如何在公開 repo 裡逐步長出來

## 定位

MiniCNN 不是要取代 PyTorch。

它比較適合用在這些情境：

- 想要一個共用的 YAML 前端介面來切不同 backend
- 想保留一條能力邊界清楚的手寫 CUDA 訓練路徑
- 想用小型 NumPy autograd stack 做學習或 framework-level 實驗
- 想原地孵化未來的 graph-based native backend，但不把未完成的東西包裝成已完成

## Backend 狀態

| Backend | 狀態 | 適合用途 |
|---|---|---|
| `torch` | 穩定 | 新模型、自訂元件、快速迭代 |
| `cuda_legacy` | 穩定但刻意狹窄 | 以固定 CIFAR-10 組態為前提的手寫 CUDA 訓練 |
| `autograd` | 穩定的教學路徑 | CPU-only 學習、可重現測試、小型框架實驗 |
| `cuda_native` | 實驗性 — 僅 forward prototype | graph IR / planner / numpy executor 研發，非正式 |

高階來看：

```text
shared YAML / CLI frontend -> torch | cuda_legacy | autograd
                               \
                                -> cuda_native [實驗] (graph IR, planner, numpy executor)
```

## 目前可以直接跑的東西

### `torch`

- 透過 flex registry 支援較廣的 `model.layers[]`
- 支援 dotted-path custom component
- 支援較完整的 scheduler、regularization 與實驗流程

### `cuda_legacy`

- `cpp/` 內的手寫 CUDA / C++ backend
- 由 `engine.backend=cuda_legacy` 的 shared-config bridge 進入
- 不支援的組合會直接 validation，不做 silent fallback
- 支援範圍刻意維持狹窄，核心仍是固定的 CIFAR-10 Conv/Pool/Linear pattern

### `autograd`

- 純 NumPy reverse-mode autodiff
- 精簡但夠用的 optimizer / layer stack
- 不依賴 torch 的教學、測試與 CPU inference 實驗

### `cuda_native`（實驗性）

以 graph-based 架構設計的實驗 backend，包含：

- 明確的 graph IR（`graph.py`, `nodes.py`）
- 嚴格驗證層（`validators.py`, `shapes.py`）
- 保守記憶體規劃（`planner.py`）
- numpy 參考 kernel 與 dispatch（`kernels.py`, `executor.py`）
- backward 原型與 SGD 訓練迴圈

支援 op：`Conv2d`, `ReLU`, `LeakyReLU`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `Linear`。

非正式 backend，僅支援 sequential graph，不取代 `cuda_legacy`。

```bash
# 查看 cuda_native 支援能力
minicnn cuda-native-capabilities

# 驗證 config 是否相容
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml

# 執行（研究用）
minicnn train-native --config configs/dual_backend_cnn.yaml train.epochs=1 dataset.num_samples=128
```

完整說明請見 [docs/cuda_native.md](docs/cuda_native.md)。

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

## 編譯 Native CUDA Library

```bash
minicnn build --legacy-make --check
```

若要同時編譯兩個 native variant：

```bash
minicnn build --legacy-make --variant both --check
```

典型輸出：

```text
cpp/libminimal_cuda_cnn_cublas.so
cpp/libminimal_cuda_cnn_handmade.so
```

native library 採 lazy-load，所以像 `minicnn --help`、`prepare-data`、
`validate-dual-config`、以及 torch-only 執行都不需要先編好 `.so`。

## 準備資料

手寫 CUDA 路徑需要先準備 CIFAR-10：

```bash
minicnn prepare-data
```

MNIST 相關 flex/autograd config 則可以用 `dataset.download=true`，
把資料下載到 `data/mnist/`。

## 常用訓練指令

訓練彈性的 torch 路徑：

```bash
minicnn train-flex --config configs/flex_cnn.yaml
```

用 shared dual-backend config 跑 torch：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
```

訓練手寫 CUDA 路徑：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

明確指定 native CUDA variant：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

訓練 NumPy autograd 路徑：

```bash
minicnn train-autograd --config configs/autograd_tiny.yaml
```

比較 backend：

```bash
minicnn compare --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

檢查目前 surface：

```bash
minicnn info
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

執行實驗性 cuda_native 路徑：

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

## Backend 邊界

專案層的 frontend 能力，比 `cuda_legacy` 本身廣很多。

這點很重要：

- `torch` 是新模型與新想法的預設落點
- `cuda_legacy` 是有 validator 與 capability boundary 的受限 backend
- `autograd` 適合學習與小型實驗
- `cuda_native` 應該作為獨立 backend 成長，而不是把 `cuda_legacy` 硬撐到無限泛用

完整支援矩陣請看 [docs/backend_capabilities.md](docs/backend_capabilities.md)，
較長期的泛用化方向請看
[docs/generalization_roadmap.md](docs/generalization_roadmap.md)。

## Config 介面

主要 shared-config 介面包含：

- `dataset`
- `model.layers`
- `train`
- `loss`
- `optimizer`
- `scheduler`
- `engine.backend`

最小範例：

```yaml
engine:
  backend: torch

dataset:
  type: cifar10
  input_shape: [3, 32, 32]
  num_classes: 10

model:
  layers:
    - type: Conv2d
      out_channels: 32
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

如果這份 config 不符合 `cuda_legacy`，請直接用
`minicnn validate-dual-config` 看具體錯誤，而不是猜測哪裡不支援。

## 可擴充性

### Custom components

Torch/flex 支援在 `model.layers[].type` 中使用 dotted-path layer factory。

例如：

```yaml
model:
  layers:
    - type: Flatten
    - type: Linear
      out_features: 32
    - type: minicnn.extensions.custom_components.ConvBNReLU
      out_channels: 32
```

詳見 [docs/custom_components.md](docs/custom_components.md)。

## 文件入口

建議先看：

- [docs/USAGE.md](docs/USAGE.md)：文件索引
- [docs/architecture.md](docs/architecture.md)：整體架構與模組地圖
- [docs/backend_capabilities.md](docs/backend_capabilities.md)：backend 支援矩陣
- [docs/custom_components.md](docs/custom_components.md)：component 擴充入口
- [docs/08_autograd.md](docs/08_autograd.md)：NumPy autograd stack
- [docs/09_feature_expansion.md](docs/09_feature_expansion.md)：擴充功能說明
- [templates/README.md](templates/README.md)：可直接修改的 template configs

若要看這個 branch 內的 `cuda_native` 規劃脈絡，工作筆記放在
`comments/cuda_native/`。

## Repository 地圖

```text
minicnn/
├── cpp/                    # 手寫 CUDA / C++ backend
├── configs/                # flex、dual、autograd 的範例 configs
├── docs/                   # 設計文件、指南與 capability docs
├── examples/               # custom torch component 範例
├── comments/cuda_native/   # branch 內的 cuda_native 規劃筆記
├── src/minicnn/
│   ├── flex/               # torch/flex frontend、registry、builder、trainer
│   ├── unified/            # shared-config dispatch 與 backend bridges
│   ├── training/           # cuda_legacy 與 autograd 訓練程式
│   ├── cuda_native/        # 實驗性的 graph/planner/executor backend 開發
│   ├── nn/ ops/ optim/     # NumPy autograd stack
│   ├── compiler/ runtime/  # tracing、optimization 與 CPU inference pipeline
│   └── core/               # native build helpers 與 ctypes CUDA binding
└── tests/                  # unit 與 smoke tests
```

## 設計哲學

- backend capability 要明講，不假裝 parity
- frontend 能共用時才共用，不做虛假的抽象
- 遇到不支援的 backend 組合就 fail fast
- 實驗性 backend work 可以公開，但不能包裝成穩定功能

## License

MIT
