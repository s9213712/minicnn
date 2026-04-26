# MiniCNN

[English README](README.md)

![status](https://img.shields.io/badge/status-beta-yellow)
![frontend](https://img.shields.io/badge/frontend-YAML%20%2B%20CLI-blue)
![native](https://img.shields.io/badge/native-CUDA-green)

MiniCNN 是一個以組態驅動的深度學習專案，用來研究「同一個前端介面」與「不同 backend 能力邊界」之間的落差。

Shell 說明：
repo 內大多數多行指令範例預設採用 Bash 的續行符號 `\`。如果你在
Windows PowerShell 執行，請改用反引號 `` ` ``；如果你用的是 `cmd.exe`，
則改用 `^`。

目前這個 repo 實際提供四種 backend 角色：

- `torch`：透過 `train-flex` / `train-dual` 扮演最廣、最穩的 reference implementation
- `cuda_native`：透過 `train-native` 扮演主要的 native backend 發展方向
- `autograd`：透過 `train-autograd` 扮演內部 correctness oracle
- `cuda_legacy`：透過 `train-dual` 保留手寫 CUDA 的歷史維護路徑

## 快速開始

如果你想從 clone 直接走到第一個成功指令，最短路徑是：

```bash
python3 -m pip install -e .
minicnn smoke
minicnn show-model --config configs/flex_cnn.yaml --format text
```

如果這三個指令都成功，代表 repo 結構、config parsing 和核心 CLI surface 都正常。接下來：

- 用 `minicnn train-flex --config configs/flex_cnn.yaml` 走最廣的 reference path
- 用 `minicnn train-autograd --config configs/autograd_tiny.yaml` 走最小的 CPU-only reference path
- 想按任務找文件時，直接把 `USAGE.md` 當成文件索引

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

- 想要一個共用的 YAML 前端介面來切不同 backend 角色
- 想先在 torch reference path 驗證新想法
- 想用小型 NumPy autograd stack 做 correctness 檢查或 framework-level 實驗
- 想在公開 repo 裡推進 native backend，但不強迫歷史路徑跟著一起膨脹

## Backend 角色

| Backend | 角色 | 目前狀態 |
|---|---|---|
| `torch` | reference implementation | 穩定，功能最廣，新模型功能優先落這裡 |
| `cuda_native` | 主要 native backend | beta、graph-based、具備 ordered DAG 能力、reference mode 加上部分 real-CUDA `gpu_native` execution |
| `autograd` | correctness oracle | 穩定、CPU-only，適合 deterministic 檢查與框架學習 |
| `cuda_legacy` | 歷史 native backend | 在窄邊界內穩定，但屬 maintenance-only，不是新功能擴充主戰場 |

## 新功能 rollout 順序

新增能力時，預設順序是：

1. 先做 `torch/flex`
2. 如果 correctness 驗證有價值，再補 `autograd`
3. 再視情況推進 `cuda_native`
4. `cuda_legacy` 只做維護必要修補，不作為預設擴充目標

高階來看：

```text
shared YAML / CLI frontend -> torch [REFERENCE] | autograd [ORACLE]
                               \
                                -> cuda_native [PRIMARY NATIVE] (beta graph IR, planner, reference + gpu_native executor)
                               \
                                -> cuda_legacy [MAINTENANCE ONLY] (historical handwritten CUDA path)
```

## 最近的工程進度

最近這輪 cleanup / refactor 已經落在 `main` 上。

目前 repo 的工程收斂狀態包括：

- CLI 已拆成 parser、readonly、training/compare 等較聚焦的 helper 模組
- `flex` 訓練路徑已拆出 context setup、run orchestration、reporting、device resolution 與 step helpers
- `cuda_native` 的 unified training 已拆成 bridge、runtime loop、support / reporting helpers
- artifact inspect/export 與 checkpoint payload handling 已移到獨立 helper 層
- `healthcheck`、`doctor`、`smoke`、`validate-*` 與 inspection 指令都有 JSON-friendly 的診斷/驗證介面
- `show-model` 與 `show-graph` 已是實際可用的 introspection 指令，不再是 placeholder
- `cuda_native` graph semantics 已從單純 sequential graph 擴成具名 tensor wiring 的 ordered DAG，並支援 `Add` / `Concat`
- `cuda_native` training surface 已擴到 `SGD`、`Adam`、`AdamW`、`RMSprop`、`CrossEntropyLoss`、`BCEWithLogitsLoss`、`MSELoss`、`label_smoothing`、`grad_accum_steps` 與 beta AMP
- `cuda_native` strict `gpu_native` 已有 real CIFAR-10 two-Conv native CUDA helper training runbook
- `summary.json` / `metrics.jsonl` 現在也會穩定輸出 planner、AMP 與 optimizer-state telemetry

結果是：對使用者來說，主要命令集合仍維持精簡；對維護者來說，模組邊界更清楚、
輸出契約更明確，backend 角色文件也和實際程式行為對齊。

## 目前可以直接跑的東西

### `torch`

- 新 frontend / layer 想法的 reference implementation
- 透過 flex registry 支援較廣的 `model.layers[]`
- 支援 dotted-path custom component
- 支援較完整的 scheduler、regularization 與實驗流程

### `cuda_legacy`

- maintenance-only 的歷史 backend
- `cpp/` 內的手寫 CUDA / C++ backend
- 由 `engine.backend=cuda_legacy` 的 shared-config bridge 進入
- 不支援的組合會直接 validation，不做 silent fallback
- 支援範圍刻意維持狹窄，核心仍是固定的 CIFAR-10 Conv/Pool/Linear pattern

### `autograd`

- 內部 correctness oracle，方便做 CPU 參考檢查
- 純 NumPy reverse-mode autodiff
- 精簡但夠用的 optimizer / layer stack
- 不依賴 torch 的教學、測試與 CPU inference 實驗

### `cuda_native`（主要 native 方向，現為 beta）

目前 repo 裡主要的 native backend 成長方向，採 graph-based 架構，包含：

- 明確的 graph IR（`graph.py`, `nodes.py`）
- 嚴格驗證層（`validators.py`, `shapes.py`）
- 保守記憶體規劃（`planner.py`）
- numpy 參考 kernel 與 dispatch（`kernels.py`, `executor.py`）
- backward 原型與 SGD 訓練迴圈
- layout 驗證（`layouts.py` — `validate_graph_layouts()`）
- 記憶體估算與 pool（`memory.py` — `memory_footprint()`、`BufferPool`）
- 觀測工具（`debug.py` — `dump_graph()`、`dump_plan()`、`TracingForwardExecutor`）

支援 op：`BatchNorm2d`（forward/backward prototype）、`Conv2d`、`DepthwiseConv2d`、`PointwiseConv2d`、`GroupNorm`、`LayerNorm`、`LayerNorm2d`、`ResidualBlock`、`ConvNeXtBlock`、`Dropout`、`DropPath`、`Add`、`Concat`、`ReLU`、`LeakyReLU`、`Sigmoid`、`Tanh`、`SiLU`、`GELU`、`Identity`、`MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`（僅 `output_size=(1,1)`）、`GlobalAvgPool2d`、`Flatten`、`Linear`。

目前通過驗證的支援範圍：

- dataset：`random`、`cifar10`、`mnist`
- loss：`CrossEntropyLoss`（可搭配 `label_smoothing`）、`BCEWithLogitsLoss`（僅 binary output）、`MSELoss`
- optimizer：支援 `SGD`、`Adam`、`AdamW`、`RMSprop`，可選 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- `train.grad_accum_steps >= 1`
- `train.amp=true|false`，帶 beta 級的 loss scaling / overflow backoff
- `summary.json` 會輸出 `amp_runtime`、`optimizer_runtime`、`planner` 與 `performance_report`
- `metrics.jsonl` 每個 epoch row 會輸出 AMP、optimizer 與 planner telemetry

這條 backend 現在已經是 beta 級，`training_stable=true`、`backward_stable=true`，但仍不適合宣稱 production-ready。預設路徑是 GPU-first 的 `gpu_native_auto`；對已支援的 subset 會優先走真正的 CUDA helper / device-pointer 路徑，`reference_numpy` 則保留成明確 fallback 與 parity baseline。它已支援具名 tensor wiring 與 `Add` merge 的 ordered DAG 執行；後續 native 功能仍應優先往這條線發展，`cuda_legacy` 則維持窄邊界維護。

目前也已有 hermetic native smoke 範本可直接用於：

- 顯式 ConvNeXt primitive 路徑：`templates/cifar10/convnext_explicit_cuda_native_smoke.yaml`
- named `ConvNeXtBlock` 路徑：`templates/cifar10/convnext_tiny_cuda_native_smoke.yaml`
- `ResidualBlock` 路徑：`templates/cifar10/resnet_like_cuda_native_smoke.yaml`

目前 strict real-CUDA CIFAR-10 training 可用於 two-Conv helper subset：

```bash
minicnn validate-cuda-native-config --config configs/cifar10_cuda_native_gpu_stronger.yaml

minicnn train-native --config configs/cifar10_cuda_native_gpu_stronger.yaml
```

目前 real-data GPU 結果與效能瓶頸請見 [docs/cuda_native_gpu_cifar10_runbook.md](docs/cuda_native_gpu_cifar10_runbook.md)。

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
後續擴充方向請見 [docs/cuda_native_expansion_plan.md](docs/cuda_native_expansion_plan.md)。
從實驗走向可承諾實裝邊界的規劃請見 [docs/cuda_native_productionization_plan.md](docs/cuda_native_productionization_plan.md)。
AMP 從 experimental 畢業到 beta 的檢查項目請見 [docs/cuda_native_amp_graduation_checklist.md](docs/cuda_native_amp_graduation_checklist.md)。
從 NumPy reference execution 走向真正 GPU execution 的未來路徑請見 [docs/cuda_native_gpu_enablement_plan.md](docs/cuda_native_gpu_enablement_plan.md)。
目前完整 CIFAR-10 strict `gpu_native` 訓練指令請見 [docs/cuda_native_gpu_cifar10_runbook.md](docs/cuda_native_gpu_cifar10_runbook.md)。

真實資料集示範：

```bash
PYTHONPATH=src python3 examples/cuda_native_amp_cifar10_beta_demo.py \
  --data-root data/cifar-10-batches-py \
  --artifacts-root /tmp/minicnn_cuda_native_beta_demo
```

## 快速開始

# Linux / macOS

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

# Windows PowerShell

```powershell
git clone https://github.com/s9213712/minicnn.git
cd minicnn
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .[dev]
minicnn smoke
pytest
```

如果 PowerShell 擋下 `.venv\Scripts\Activate.ps1`，先在同一個 shell 執行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

如果 `minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch`
在 Windows 上仍然沒有吃到 GPU，先確認目前環境裝到的真的是 CUDA 版
PyTorch wheel：

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

`minicnn smoke` 是安裝後最推薦先跑的自檢。它會檢查 repo 結構、解析內建
config、跑一次小型 compiler trace，並驗證 `cuda_legacy` 與
`cuda_native` 的 config 驗證邊界。

## 最小依賴矩陣

| 指令 / 功能 | 需要 PyTorch | 需要 native library（`.so`/`.dll`） | 需要 CIFAR-10 data |
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

native library 採 lazy-load，所以像 `minicnn --help`、`prepare-data`、
`validate-dual-config`、以及 torch-only 執行都不需要先編好 `.so`。

Windows 路徑請看已手動驗證的
[docs/guide_windows_build.md](docs/guide_windows_build.md)。目前記錄下來的
腳本流程會把 DLL 輸出到 `cpp\Release\`。

如果只想先確認 ctypes 載入是否正常，可直接跑：

```bash
python3 -u examples/mnist_ctypes/check_native_library.py --variant handmade
```

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

Windows 使用者若想先驗證 DLL 可載入，再開始訓練，可直接跑：

```powershell
python -u examples\mnist_ctypes\check_native_library.py --variant handmade
python -u examples\mnist_ctypes\check_native_library.py --path cpp\minimal_cuda_cnn_cublas.dll
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
minicnn evaluate-checkpoint --config configs/dual_backend_cnn.yaml \
  --summary artifacts/example-run/summary.json
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
minicnn evaluate-checkpoint --config configs/dual_backend_cnn.yaml \
  --summary artifacts/example-run/summary.json
minicnn export-torch-checkpoint --path artifacts/models/example_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/example_autograd_export.pt
```

這樣可以同時快速看 schema，也能直接重跑 torch/flex checkpoint 的測試集準確率。

若要拿真實圖片做推理，可用 repo 內的正式範例。它會先把大圖裁切/縮放成 config
支援的輸入尺寸，再做 top-k 預測：

```bash
python -u examples/inference/predict_image.py \
  --config configs/dual_backend_cnn.yaml \
  --summary artifacts/example-run/summary.json \
  --image path/to/photo.jpg \
  --topk 5
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
- `torch/flex` 是 reference implementation，也是新功能第一站
- `autograd` 是內部 correctness oracle
- `cuda_native` 是主要 native 方向，但仍屬實驗性
- `cuda_legacy` 保留在 validator 定義的邊界內做維護，不是新功能預設成長點

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
- [docs/cuda_native.md](docs/cuda_native.md)：主要 native backend 指南
- [docs/custom_components.md](docs/custom_components.md)：dotted-path 元件擴展
- [docs/model_artifacts.md](docs/model_artifacts.md)：checkpoint 格式、復用邊界與示範
- [templates/README.md](templates/README.md)：可直接修改的 template config
- [examples/README.md](examples/README.md)：canonical example path 與可執行範例總覽

`docs/` 內也保留了一些背景報告與歷史比較文件；現在 [USAGE.md](USAGE.md)
會把「目前仍是操作主線的文件」和「歷史／報告文件」分開，不再混成同一層。

## Repository 目錄結構

```text
minicnn/
├── cpp/                    # 手寫 CUDA / C++ backend
├── configs/                # flex、dual、autograd 路徑的範例 config
├── docs/                   # 設計說明、指南與 capability 文件
├── examples/               # 可執行範例、inference demo 與 native ctypes 樣例
├── src/minicnn/
│   ├── flex/               # torch/flex 前端、registry、builder、trainer
│   ├── unified/            # shared-config dispatch 與 backend bridge
│   ├── training/           # cuda_legacy 與 autograd 訓練程式碼
│   ├── cuda_native/        # graph/planner/executor backend（含 reference 與 gpu_native）
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

## CUDA Native 維護備註

- 完整 CIFAR-10 GPU-native smoke/benchmark 建議從 `configs/cifar10_cuda_native_gpu_stronger.yaml` 與 `docs/cuda_native_gpu_cifar10_runbook.md` 開始。
- `cuda_native` 對符合條件的 `gpu_native` 路徑已改成 GPU-first；NumPy reference 保留作為 fallback 與 parity 基準。
- 肥檔清理狀態記錄在 `docs/cuda_native_large_file_inventory.md`。測試檔目前刻意保留，因為它們承載 CUDA-native regression coverage。
