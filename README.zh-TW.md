# MiniCNN

[English README](README.md)

![status](https://img.shields.io/badge/status-experimental-orange)
![frontend](https://img.shields.io/badge/frontend-YAML%20%2B%20CLI-blue)
![native](https://img.shields.io/badge/native-CUDA-green)

MiniCNN 是一個以組態驅動的深度學習專案，用來研究「同一個前端介面」與「不同 backend 能力邊界」之間的落差。

目前這個 repo 實際提供四條可用路徑：

- `torch`：透過 `train-flex` / `train-dual` 做較廣泛的模型實驗
- `cuda_legacy`：透過 `train-dual` 使用手寫 CUDA 的 CIFAR-10 訓練路徑
- `autograd`：透過 `train-autograd` 使用純 NumPy 的教學型 autograd stack
- `cuda_native`：透過 `train-native` 使用實驗性 graph-based backend（非正式）

## 為什麼有這個專案

大多數框架會刻意把 kernel orchestration、記憶體處理、backend 邊界包在平滑的 API 後面。

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
| `cuda_native` | 實驗性研究 prototype | graph IR / planner / numpy executor 研發，非正式 |

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
- layout 驗證（`layouts.py` — `validate_graph_layouts()`）
- 記憶體估算與 pool（`memory.py` — `memory_footprint()`、`BufferPool`）
- 觀測工具（`debug.py` — `dump_graph()`、`dump_plan()`、`TracingForwardExecutor`）

支援 op：`BatchNorm2d`（forward/backward prototype）、`Conv2d`、`ReLU`、`LeakyReLU`、`Sigmoid`、`Tanh`、`SiLU`、`MaxPool2d`、`AvgPool2d`、`Flatten`、`Linear`。

目前通過驗證的支援範圍：

- dataset：`random`、`cifar10`、`mnist`
- loss：`CrossEntropyLoss`、`MSELoss`
- optimizer：支援 `SGD`，可選 momentum 與 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- `train.amp=false`、`train.grad_accum_steps=1`

雖然已經有 backward 與 training prototype，但這條 backend 仍屬實驗性、只支援 sequential graph，也不取代 `cuda_legacy`。

```bash
# 查看 cuda_native 支援能力
minicnn cuda-native-capabilities

# 驗證 config 是否相容
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5

# 執行（研究用）
minicnn train-native --config configs/dual_backend_cnn.yaml \
  dataset.type=random dataset.num_samples=128 dataset.val_samples=32 \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5
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
minicnn smoke
pytest
```

`minicnn smoke` 是安裝後最推薦先跑的自檢。它會檢查 repo 結構、解析內建
config、跑一次小型 compiler trace，並驗證 `cuda_legacy` 與
`cuda_native` 的 config 驗證邊界。

## 最小依賴矩陣

| 指令 / 功能 | 需要 PyTorch | 需要 native `.so` | 需要 CIFAR-10 data |
|---|---:|---:|---:|
| `minicnn --help` | 否 | 否 | 否 |
| `minicnn validate-dual-config` | 否 | 否 | 否 |
| `minicnn show-cuda-mapping` | 否 | 否 | 否 |
| `minicnn show-model` | 否 | 否 | 否 |
| `minicnn show-graph` | 否 | 否 | 否 |
| `minicnn compile` | 否 | 否 | 否 |
| `minicnn train-flex` | 是 | 否 | 視 dataset 而定 |
| `minicnn train-dual engine.backend=torch` | 是 | 否 | 視 dataset 而定 |
| `minicnn train-dual engine.backend=cuda_legacy` | 否 | 是 | 是 |
| `minicnn train-autograd` | 否 | 否 | 視 dataset 而定 |
| `minicnn train-native` | 否 | 否 | 視 dataset 而定 |

若某個指令需要 PyTorch，CLI 現在會輸出簡短且可操作的依賴訊息，不再在 import
階段直接丟 traceback。

config 或 override 寫錯時，也會以簡短訊息和 exit code `2` 失敗，而不是吐出
Python traceback。`healthcheck`、`doctor`、`smoke`、`validate-*`、
`show-cuda-mapping`、`show-model`、`show-graph`、`inspect-checkpoint` 現在都會輸出 JSON-friendly
結果；若目前 CUDA 不可用，`train.device=cuda` 也會提早失敗並提示改用
`train.device=auto` 或 `train.device=cpu`。

這些診斷、檢查與驗證命令也支援 `--format text`，方便直接在終端查看：

```bash
minicnn healthcheck --format json
minicnn doctor --format text
minicnn smoke --format json
minicnn validate-dual-config --format text
minicnn show-model --config configs/flex_cnn.yaml --format text
minicnn show-graph --config configs/flex_cnn.yaml --format json
minicnn inspect-checkpoint --path artifacts/models/example_best.pt --format text
```

`show-model` 會停留在前端/config 視角，保留 composite layer 名稱。
`show-graph` 會顯示 compiler trace 之後、經過基本 optimizer pass 的 primitive graph。

## Repo-First 資源模型

MiniCNN 目前仍以 repo checkout 為主要使用模型。像
`configs/flex_cnn.yaml`、`configs/dual_backend_cnn.yaml` 這些內建 config，
必要時會自動以 project root 為基準解析，但它們還不是完整的安裝包內建資源。

這代表：

- 主要支援 workflow 仍是 repo 內的 editable install
- 真正可攜的內建內容是 `config-template` 與 `dual-config-template`
- 若要做完整 packaged toolchain，請顯式傳入 config 路徑，不要假設 repo 檔案會出現在 site-packages
- 若內建 config 仍然找不到，CLI 現在會明確提示改傳顯式 `--config` 路徑，而不是吐 traceback

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

native library 採 lazy-load，所以像 `minicnn --help`、`prepare-data`、`validate-dual-config`、以及 torch-only 執行都不需要先編好 `.so`。

## 準備資料

手寫 CUDA 路徑需要先準備 CIFAR-10：

```bash
minicnn prepare-data
```

MNIST 相關 flex/autograd config 則可以用 `dataset.download=true`，把資料下載到 `data/mnist/`。

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
minicnn smoke
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

像 `configs/flex_cnn.yaml`、`configs/dual_backend_cnn.yaml` 這類內建 config
路徑，現在必要時會自動以 project root 為基準解析，所以不一定要在 repo
root 下執行 CLI。不過這仍屬 repo-first 便利機制，不是完整 packaged-resource
系統。

## 模型產物

各 backend 的模型檔格式目前**沒有統一**。

- torch 路徑存的是帶 `model_state` 的 `.pt` checkpoint
- autograd 存的是 `.npz` state dict arrays
- `cuda_native` 存的是平坦 `.npz` parameter dict
- `cuda_legacy` 存的是手寫 runtime checkpoint `.npz`

要找最佳模型，先看 `summary.json` 裡的 `best_model_path`。
若要快速看 schema，可直接用：

```bash
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn export-torch-checkpoint --path artifacts/models/example_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/example_autograd_export.pt
```

完整格式與復用說明見 [docs/model_artifacts.md](docs/model_artifacts.md)。

執行實驗性 cuda_native 路徑：

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

## Backend 邊界

專案層的 frontend 能力，比 `cuda_legacy` 本身廣很多。

這個區別很重要：

- `torch` 是新模型想法的預設家
- `cuda_legacy` 是有 validator 強制限制的 backend
- `autograd` 用於學習和精簡實驗
- `cuda_native` 應該作為獨立 backend 成長，而不是假裝 `cuda_legacy` 能無限延伸

完整支援矩陣見 [docs/backend_capabilities.md](docs/backend_capabilities.md)，長期方向見 [docs/generalization_roadmap.md](docs/generalization_roadmap.md)。

## Config 介面

主要的 shared-config 介面：

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

如果同一份 config 不符合 `cuda_legacy`，用 `minicnn validate-dual-config` 查看確切的相容性錯誤，而不是猜測。

## 擴展性

### 自訂元件

Torch/flex 在 `model.layers[].type` 接受 dotted-path layer factory。

範例：

```yaml
model:
  layers:
    - type: Flatten
    - type: Linear
      out_features: 32
    - type: minicnn.extensions.custom_components.ConvBNReLU
      out_channels: 32
```

見 [docs/custom_components.md](docs/custom_components.md)。

## 文件索引

從這裡開始：

- [USAGE.md](USAGE.md)：完整文件導覽與建議閱讀順序
- [docs/architecture.md](docs/architecture.md)：整體架構與模組圖
- [docs/backend_capabilities.md](docs/backend_capabilities.md)：Backend 支援矩陣
- [docs/dual_backend_guide.md](docs/dual_backend_guide.md)：shared-config routing 與 backend 邊界
- [docs/cuda_native.md](docs/cuda_native.md)：實驗性 `cuda_native` 指南
- [docs/custom_components.md](docs/custom_components.md)：dotted-path 元件擴展
- [docs/model_artifacts.md](docs/model_artifacts.md)：checkpoint 格式、復用邊界與示範
- [templates/README.md](templates/README.md)：可直接修改的 template config

`docs/` 內也保留了一些背景報告與歷史比較文件；現在 [USAGE.md](USAGE.md)
會把「目前仍是操作主線的文件」和「歷史／報告文件」分開，不再混成同一層。

## Repository 目錄結構

```text
minicnn/
├── cpp/                    # 手寫 CUDA / C++ backend
├── configs/                # flex、dual、autograd 路徑的範例 config
├── docs/                   # 設計說明、指南與 capability 文件
├── examples/               # 自訂 torch 元件範例
├── src/minicnn/
│   ├── flex/               # torch/flex 前端、registry、builder、trainer
│   ├── unified/            # shared-config dispatch 與 backend bridge
│   ├── training/           # cuda_legacy 與 autograd 訓練程式碼
│   ├── cuda_native/        # 實驗性 graph/planner/executor backend
│   ├── nn/ ops/ optim/     # NumPy autograd stack
│   ├── compiler/ runtime/  # tracing、optimization 與 CPU inference 流水線
│   └── core/               # native build helpers 與 ctypes CUDA binding
└── tests/                  # 單元測試與 smoke test
```

## 設計哲學

- 顯式的 backend 能力邊界，而非模糊的對等性宣稱
- 一個前端介面，在這個範圍內抽象是誠實的
- 不支援的 backend 組合快速失敗
- 實驗性 backend 工作維持可見，但不偽裝成穩定

## License

MIT
