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

若要同時編譯兩種 native variant 方便比較：

```bash
minicnn build --legacy-make --variant both --check
```

會產生：

```text
cpp/libminimal_cuda_cnn_cublas.so
cpp/libminimal_cuda_cnn_handmade.so
```

Windows 可用 PowerShell 編譯 DLL：

```powershell
.\scripts\build_windows_native.ps1 -Variant both
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

手寫 CUDA backend 可以指定載入哪個 `.so` variant：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

native library 會 lazy-load，所以 `--help`、`prepare-data`、`validate-dual-config`、torch backend 等非 CUDA 指令不需要先編好 `.so`。

快速 debug 可直接使用 config override：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas \
  train.epochs=1 train.batch_size=32 \
  dataset.num_samples=128 dataset.val_samples=32
```

legacy trainer 也支援常用環境變數覆蓋，例如 `MINICNN_EPOCHS`、`MINICNN_BATCH`、`MINICNN_N_TRAIN`、`MINICNN_N_VAL`。

訓練產生的最佳模型檔固定寫入：

```text
src/minicnn/training/models/
```

PyTorch backend 會寫入 `*_best.pt`；CUDA legacy backend 會寫入 `*_best_model_split.npz`。每次實驗的 metrics 與 summary 仍保留在 `artifacts/`。

## 本機 Smoke Test 結果

2026-04-19 使用 RTX 3050 Laptop GPU 驗證。本次最新快速驗證使用 `features/backend-smoke-matrix/run_smoke_matrix.py`，`128` 筆 train、`32` 筆 validation，batch size `32`，訓練 `1` epoch：

| Backend | Native variant | Train acc | Val acc | Test acc | Epoch time |
|---|---|---:|---:|---:|---:|
| `torch` | PyTorch CUDA | `11.72%` | `0.00%` | `10.36%` | `1.3s` |
| `cuda_legacy` | `cublas` | `7.03%` | `6.25%` | `12.97%` | `0.2s` |
| `cuda_legacy` | `handmade` | `7.03%` | `6.25%` | `12.97%` | `0.2s` |

Smoke test 的模型檔會寫入 `src/minicnn/training/models/`；若照 docs 的指令執行，run logs 會在 `/tmp/minicnn_backend_compare`。

驗證也包含 `pytest`、沒有 native `.so` 時的 CLI help、config validation、Python compile checks，以及 `minicnn build --legacy-make --variant both --check`。

## 為什麼有兩個 backend

本專案刻意保留兩種 workflow：

- **Torch backend**：支援較廣的 layer、快速實驗、自訂 dotted-path component。
- **CUDA backend**：保留手寫 CUDA CNN 路徑，方便低階控制與 backend ownership。

## MiniCNN autograd core

MiniCNN 也有自己的小型 CPU/NumPy autograd stack，位置在 `src/minicnn/nn/tensor.py`、`src/minicnn/ops/` 與 `src/minicnn/nn/layers.py`。它支援 `Tensor.backward()` 的 reverse-mode autodiff，涵蓋 scalar/tensor arithmetic、broadcasting、matrix multiply、reductions、reshape、ReLU、`log_softmax`、`cross_entropy`、可訓練的 `Parameter`、不依賴 torch 的輕量 `SGD`/`Adam` optimizer，以及 `Linear`、`Conv2d`、`MaxPool2d`、`BatchNorm2d`、`Flatten`、`ResidualBlock` 等小型教學 layer。

這個 core 主要用於 framework-level 測試與小型教學範例。Torch backend 仍使用 PyTorch autograd；手寫 CUDA backend 仍使用明確寫在 CUDA/C++ 裡的 backward kernels。

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
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn train --config configs/dual_backend_cnn.yaml engine.backend=torch train.epochs=1
minicnn train-torch --config configs/dual_backend_cnn.yaml train.epochs=1
minicnn train-cuda --config configs/dual_backend_cnn.yaml train.epochs=1
minicnn train-autograd --config configs/autograd_tiny.yaml train.epochs=1
minicnn compare --config configs/dual_backend_cnn.yaml train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
minicnn dual-config-template
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn validate-config --config configs/dual_backend_cnn.yaml
minicnn compile --config configs/autograd_tiny.yaml
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
minicnn/
├── configs/
│   ├── dual_backend_cnn.yaml          # 主要 CIFAR-10 config；在這切 torch/cuda_legacy
│   ├── dual_backend_torch_custom.yaml # 自訂 dotted-path component 範例
│   ├── autograd_tiny.yaml             # CPU/NumPy autograd 小型 smoke config
│   ├── flex_*.yaml                    # PyTorch flex trainer 範例
│   ├── train_cuda.yaml                # legacy CUDA compatibility config
│   └── train_torch.yaml               # Torch baseline compatibility config
├── cpp/
│   ├── Makefile                       # Linux native .so build，含 cublas/handmade variant
│   ├── CMakeLists.txt                 # Linux/Windows helper 使用的 CMake build path
│   ├── include/                       # native public headers
│   └── src/                           # CUDA/C++ kernels 與 C API 實作
├── docs/                              # 教學與 reference docs
├── examples/                          # 自訂 PyTorch component 範例
├── features/
│   ├── README.md                      # 隔離 prototype 的規則
│   └── backend-smoke-matrix/          # 比較 torch/cublas/handmade smoke runs 的範例 feature
├── scripts/
│   └── build_windows_native.ps1       # Windows CUDA DLL build helper
├── src/minicnn/
│   ├── cli.py                         # minicnn command entrypoint
│   ├── autograd/                      # Tensor、Function、Context compatibility namespace
│   ├── compiler/                      # MiniCNN IR、tracer、passes、scheduler、lowering 邊界
│   ├── config/                        # config schema、loader、legacy settings bridge
│   ├── core/                          # native build helper 與 lazy ctypes CUDA binding
│   ├── data/                          # CIFAR-10 準備/載入
│   ├── flex/                          # config-driven PyTorch model builder 與 trainer
│   ├── models/                        # CPU/NumPy model registry 與 config builder
│   ├── nn/                            # MiniCNN Tensor、Parameter 與 CPU/NumPy autograd core
│   ├── ops/                           # CPU/NumPy differentiable layer ops
│   ├── optim/                         # MiniCNN SGD 與 Adam optimizer
│   ├── runtime/                       # graph executor、backend protocol、memory pool、profiler
│   ├── training/
│   │   ├── train_cuda.py              # legacy CUDA CIFAR-10 training 入口
│   │   ├── train_torch_baseline.py    # PyTorch baseline training 入口
│   │   ├── train_autograd.py          # CPU/NumPy autograd training 入口
│   │   ├── models/                    # checkpoint 固定輸出資料夾
│   │   ├── cuda_ops.py                # CUDA copy/layout/forward helper wrapper
│   │   ├── cuda_workspace.py          # 每個 batch 重用的 GPU workspace
│   │   ├── evaluation.py              # CUDA eval forward/accuracy helper
│   │   └── checkpoints.py             # CUDA checkpoint save/load/free helper
│   └── unified/
│       ├── config.py                  # shared default config 與 override merge
│       ├── cuda_legacy.py             # 將 shared config 映射到 legacy CUDA settings
│       └── trainer.py                 # 將 train-dual dispatch 到 torch 或 cuda_legacy
└── tests/                             # config、import、framework wiring 的 unit/smoke tests
```

主要資料夾與檔案用途：

| Path | 用途 |
|---|---|
| `configs/` | torch、cuda_legacy、flex、自訂 component、AlexNet-like、ResNet-like 訓練用 YAML config。 |
| `configs/autograd_tiny.yaml` | CPU/NumPy autograd trainer 的小型 random-data config。 |
| `cpp/` | 原生 CUDA/C++ source、header、Makefile、CMake build 檔。 |
| `cpp/src/cublas_context.cu` | forward/backward CUDA code 共用的 cuBLAS handle。 |
| `cpp/src/core.cu` | GEMM forward 路徑；用 `USE_CUBLAS` 切換 cuBLAS 或手寫 CUDA。 |
| `cpp/src/conv_backward.cu` | convolution backward kernel 與 cuBLAS/handmade weight-gradient 路徑。 |
| `docs/` | 編譯、C API、Python ctypes、C++ linking、layout/debug、Windows build 教學。 |
| `examples/` | 最小自訂 PyTorch component 範例。 |
| `features/` | 隔離原型區；正式 production code 預設不應 import 這裡，內含 `backend-smoke-matrix/` 作為範例 feature。 |
| `scripts/build_windows_native.ps1` | Windows CUDA DLL variant 的 PowerShell build helper。 |
| `src/minicnn/cli.py` | 主要 CLI entrypoint。 |
| `src/minicnn/autograd/` | MiniCNN `Tensor`、`Parameter`、`Function`、`Context`、`no_grad`、`backward` 的 compatibility namespace。 |
| `src/minicnn/compiler/` | 輕量 IR、model-config tracer、optimizer passes、scheduler，以及明確標示的 lowering 邊界。 |
| `src/minicnn/core/build.py` | `minicnn build` 使用的 native build/check helper。 |
| `src/minicnn/core/cuda_backend.py` | 原生 CUDA library 的 lazy ctypes loader 與 Python helper。 |
| `src/minicnn/core/fused_ops.py` | Conv2d + BatchNorm2d + ReLU fusion 語意的 NumPy reference helper。 |
| `src/minicnn/data/` | CIFAR-10 下載/載入與 random dataset helper。 |
| `src/minicnn/flex/` | PyTorch config-driven model/loss/optimizer/scheduler builder 與 trainer；包含 torch-only `ResidualBlock` 與 `GlobalAvgPool2d`。 |
| `src/minicnn/models/` | CPU/NumPy MiniCNN model registry、shape inference、config builder、graph helper。 |
| `src/minicnn/nn/` | MiniCNN framework layer：`Module`、`Sequential`、`Tensor`、`Parameter` 與 CPU/NumPy autograd functions。 |
| `src/minicnn/nn/tensor.py` | reverse-mode autograd engine，支援 scalar/tensor ops、broadcasting、matmul、reductions、ReLU、`log_softmax`、`cross_entropy`。 |
| `src/minicnn/nn/layers.py` | CPU/NumPy MiniCNN layers：`Linear`、`Conv2d`、`MaxPool2d`、`BatchNorm2d`、`Flatten`、`ReLU`、`ResidualBlock`。 |
| `src/minicnn/ops/` | MiniCNN layers 使用的 differentiable NumPy ops。 |
| `src/minicnn/optim/` | 輕量 optimizer 介面；`SGD` 與 `Adam` 可更新 MiniCNN `Parameter`，不需要 torch。 |
| `src/minicnn/runtime/` | 小型 graph executor、backend protocol、tensor memory pool、profiler utilities。 |
| `src/minicnn/training/train_cuda.py` | legacy CUDA CIFAR-10 training loop 入口。 |
| `src/minicnn/training/train_autograd.py` | random-data CPU/NumPy autograd training loop，輸出 `*_autograd_best.npz`。 |
| `src/minicnn/training/models/` | 最佳模型 checkpoint 固定輸出位置；產生的 `*.pt` 與 `*.npz` 檔不進 git。 |
| `src/minicnn/training/cuda_ops.py` | legacy training loop 使用的小型 CUDA operation wrapper。 |
| `src/minicnn/training/cuda_workspace.py` | 可重用 batch GPU workspace，含 double-free 保護。 |
| `src/minicnn/training/evaluation.py` | CUDA evaluation forward path 與 accuracy helper。 |
| `src/minicnn/training/checkpoints.py` | CUDA checkpoint save/reload 與 GPU pointer cleanup。 |
| `src/minicnn/training/train_torch_baseline.py` | 對齊手寫 CUDA update 規則的 PyTorch baseline。 |
| `src/minicnn/unified/` | shared config compiler 與 `torch`/`cuda_legacy` dispatcher。 |
| `tests/` | 不需 GPU 的 unit/smoke tests；真正 GPU 訓練用 CLI 指令另跑。 |

Windows native build 說明在 [docs/07_windows_build.md](docs/07_windows_build.md)。

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
