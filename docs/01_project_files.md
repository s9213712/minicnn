# 專案檔案說明

本文整理 `cpp/include` 與 `cpp/src` 中各檔案的責任。預設訓練流程主要透過 `extern "C"` 匯出的 C API 呼叫 `.so`，C++ layer 類別是 secondary API，供 C++ 端範例與實驗直接使用。

## 目錄結構

```text
minicnn/
├── cpp/
│   ├── Makefile
│   ├── include/
│   │   ├── cuda_check.h
│   │   ├── dense_layer.h
│   │   ├── network.h
│   │   └── tensor.h
│   ├── src/
│   │   ├── core.cu
│   │   ├── cublas_context.cu
│   │   ├── memory.cu
│   │   ├── loss_layer.cu
│   │   ├── conv_backward.cu
│   │   ├── dense_layer.cu
│   │   ├── optimizer.cu
│   │   ├── layout_convert.cu
│   │   ├── maxpool_store.cu
│   │   ├── maxpool_backward_use_idx.cu
│   │   └── ...
│   ├── libminimal_cuda_cnn.so
│   ├── libminimal_cuda_cnn_cublas.so
│   └── libminimal_cuda_cnn_handmade.so
├── src/minicnn/
│   ├── autograd/
│   ├── cli.py
│   ├── compiler/
│   ├── core/
│   ├── cuda_native/
│   ├── data/
│   ├── flex/
│   ├── models/
│   ├── nn/
│   ├── ops/
│   ├── optim/
│   ├── runtime/
│   ├── training/
│   │   ├── cuda_batch.py
│   │   ├── legacy_data.py
│   │   ├── loop.py
│   │   ├── train_cuda.py
│   │   └── train_torch_baseline.py
│   └── unified/
├── configs/
├── tests/
└── docs/
```

`data/cifar-10-batches-py/`、`cpp/*.so`、`artifacts/models/*`、`__pycache__/`、`.pytest_cache/` 與其他 runtime artifacts 都是本機檔案，不屬於 Git 版本內容。

## include

| 檔案 | 作用 |
|---|---|
| `cuda_check.h` | CUDA 錯誤檢查工具。`CUDA_CHECK(expr)` 檢查 runtime API 回傳值；release 版 `CUDA_KERNEL_CHECK()` 只檢查 launch error，debug build 啟用 `MINICNN_DEBUG_SYNC` 後才同步 GPU。 |
| `cublas_check.h` | cuBLAS 錯誤檢查工具，集中供 `core.cu` 與 `conv_backward.cu` 共用。 |
| `tensor.h` | Secondary C++ API 的 `CudaTensor` RAII 包裝，管理 GPU tensor 記憶體，提供 host/device copy。 |
| `network.h` | Secondary C++ layer 介面與 `ConvLayer`、`ReLULayer`、`MaxPoolLayer` 宣告；forward output 以 `std::unique_ptr<CudaTensor>` 表示所有權。 |
| `dense_layer.h` | Secondary C++ API 的 `DenseLayer` 宣告，forward output 以 RAII pointer 管理。 |

主要 Python/CLI 訓練路徑使用 flat C ABI 和 `ctypes`。`network.h`、`dense_layer.h`、`tensor.h` 保留給 [05_cpp_linking.md](05_cpp_linking.md) 的 C++ 使用場景，不是預設訓練入口。

## src

| 檔案 | 作用 |
|---|---|
| `memory.cu` | 匯出 `gpu_malloc`、`gpu_free`、`gpu_memcpy_h2d`、`gpu_memcpy_d2h`、`gpu_memset`，供 Python/C++ 管理 GPU 記憶體。 |
| `core.cu` | 基礎 forward kernel：`im2col_forward`、`gemm_forward`、`apply_relu`、`apply_maxpool`。`USE_CUBLAS=1` 時 `gemm_forward` 使用 cuBLAS；`USE_CUBLAS=0` 時使用手寫 GEMM kernel。 |
| `cublas_context.cu` | 集中建立並重用單一 cuBLAS handle，供 forward GEMM 與 convolution backward 共用。 |
| `backward.cu` | ReLU backward 與不保存 index 的 NCHW maxpool backward。 |
| `conv_backward.cu` | 卷積層 backward：`USE_CUBLAS=1` 時 weight gradient 使用 im2col + cuBLAS GEMM；`USE_CUBLAS=0` 時保留手寫 CUDA fallback。input gradient 仍使用直接 CUDA kernel。訓練主流程使用 `conv_backward_precol` 重用 forward im2col buffer。 |
| `dense_layer.cu` | 全連接層 forward/backward：`dense_forward`、`dense_backward_full`。 |
| `loss_layer.cu` | `softmax_forward`、`softmax_backward`、`softmax_xent_grad_loss_acc`、`count_correct` 與 `gemm_backward`。訓練主流程使用 fused softmax cross-entropy kernel，一次產生 loss sum、accuracy count、probabilities 與 logits gradient。 |
| `optimizer.cu` | Optimizer kernel。`apply_sgd_update` 執行純 SGD；`apply_momentum_update` 執行 Momentum SGD；`conv_update_fused` 在 GPU 端合併 weight decay、gradient clipping、momentum update；`clip_inplace` 做 GPU in-place gradient clipping。 |
| `layout_convert.cu` | `nchw_to_cnhw` 與 `cnhw_to_nchw`。部分 kernel 輸出以 CNHW 儲存，訓練時常需要轉換。 |
| `reorganize.cu` / `reorganize_backward.cu` | 舊版 layout 重排 API。新程式建議優先用 `layout_convert.cu` 的明確 NCHW/CNHW 函式。 |
| `maxpool_store.cu` | 帶 max index 的 maxpool forward：`maxpool_forward_store`。 |
| `maxpool_backward_use_idx.cu` | 搭配 `maxpool_forward_store` 做 maxpool backward。 |
| `maxpool_backward_nchw.cu` | NCHW maxpool backward 版本。保留舊 `void maxpool_backward_nchw(...)` ABI，並提供 `int maxpool_backward_nchw_status(...)` 讓 Python wrapper 可把參數錯誤轉成可捕捉例外。 |
| `leaky_relu.cu` | LeakyReLU forward/backward，含 CNHW 與 NCHW 命名版本。 |
| `layer_norm.cu` | LayerNorm forward/backward。 |
| `network.cu` | C++ layer 類別的 forward 實作，供 C++ 端使用。`ConvLayer` 重用 im2col cache，ReLU 用 out-of-place kernel 避免 D2D copy。 |
| `gpu_monitor.cu` | `check_gpu_status()`，透過 CUDA runtime `cudaMemGetInfo` 印出 `used_bytes,total_bytes`，不啟動 shell subprocess。 |

## Python package

| 檔案 | 作用 |
|---|---|
| `src/minicnn/cli.py` | `minicnn` CLI entrypoint，提供 build、prepare-data、train-flex、train-dual、train-autograd、compare、validate-config、compile 等命令。 |
| `src/minicnn/autograd/` | `Tensor`、`Parameter`、`Function`、`Context`、`no_grad` 與 `backward` compatibility namespace。 |
| `src/minicnn/compiler/` | 輕量 MiniCNN IR、config tracer 與 fusion/cleanup pass；目前 `compile` 指令停在 IR summary，不做獨立 lowering 階段。 |
| `src/minicnn/config/parsing.py` | CLI/config scalar parser、strict boolean parser、以及支援 list-index 的 dotted override 寫入 helper。 |
| `src/minicnn/core/build.py` | native CUDA shared library build/check wrapper，支援 default、cublas、handmade、both variants。 |
| `src/minicnn/core/cuda_backend.py` | native CUDA library 的 lazy `ctypes` loader；非 CUDA 指令 import 時不會載入 `.so`。`reset_library_cache()` 供同一 process 切換 native variant 時清掉舊 handle。 |
| `src/minicnn/core/fused_ops.py` | Conv2d + BatchNorm2d + ReLU fusion 語意的 NumPy reference helper。 |
| `src/minicnn/cuda_native/` | 實驗性的 native graph / planner / executor backend；已經有公開 CLI 入口，但仍屬 research prototype。 |
| `src/minicnn/data/` | CIFAR-10 與 MNIST 準備/資料載入。 |
| `src/minicnn/flex/` | PyTorch flexible config-driven model builder、registry、trainer。 |
| `src/minicnn/models/` | CPU/NumPy MiniCNN model registry、shape inference、config builder 與 graph helper。 |
| `src/minicnn/nn/` | MiniCNN framework layer，包含 `Module`、`Sequential`、`Tensor`、`Parameter` 與 CPU/NumPy autograd functions。 |
| `src/minicnn/nn/tensor.py` | reverse-mode autograd engine；支援 arithmetic、broadcasting、matmul、reductions、reshape、ReLU、`log_softmax`、`cross_entropy` 與 `Tensor.backward()`。 |
| `src/minicnn/nn/layers.py` | CPU/NumPy MiniCNN layers：`Linear`、`Conv2d`、`MaxPool2d`、`BatchNorm2d`、`Flatten`、`ReLU`、`ResidualBlock`。 |
| `src/minicnn/ops/` | MiniCNN layers 使用的 differentiable NumPy ops。 |
| `src/minicnn/optim/` | 輕量 optimizer 介面；`SGD` 與 `Adam` 可在不依賴 torch 的情況下更新 MiniCNN `Parameter`。 |
| `src/minicnn/runtime/` | 小型 graph executor、tensor memory pool、profiler utilities 與 CPU inference pipeline。 |
| `src/minicnn/unified/` | shared config compiler，將支援的 config 映射到 `torch`、`cuda_legacy` 或實驗性的 `cuda_native` backend。 |
| `src/minicnn/training/train_cuda.py` | legacy CUDA CIFAR-10 orchestration 入口：資料、epoch、validation、checkpoint、LR reduction、early stop、final test evaluation。 |
| `src/minicnn/training/cuda_batch.py` | CUDA batch 級 forward/loss/backward/update 步驟。`train_cuda.py` 呼叫這裡，避免訓練控制流程混入 kernel orchestration 細節。 |
| `src/minicnn/training/train_autograd.py` | random-data CPU/NumPy autograd training loop，輸出 `*_autograd_best.npz`。 |
| `src/minicnn/training/loop.py` | 共用訓練狀態與格式化 helper：`RunningMetrics`、`LrState`、`FitState`、`EpochTimer`、LR plateau reduction、epoch summary。 |
| `src/minicnn/training/legacy_data.py` | legacy CUDA 與 Torch baseline 共用的 CIFAR-10 load/normalize helper。 |
| `src/minicnn/training/cuda_ops.py` | legacy loop 使用的 CUDA copy、layout、forward wrapper。`maxpool_backward_nchw_into()` 優先使用 status-returning native API。 |
| `src/minicnn/training/cuda_workspace.py` | batch GPU workspace，重用每 batch buffer 並保護 double-free。 |
| `src/minicnn/training/evaluation.py` | CUDA evaluation forward path 與 accuracy helper。 |
| `src/minicnn/training/checkpoints.py` | CUDA checkpoint save/reload 與 GPU pointer cleanup。 |
| `src/minicnn/training/train_torch_baseline.py` | 對齊 CUDA update 規則的 PyTorch baseline orchestration 與 batch helper。 |
| `artifacts/models/` | 最佳模型輸出位置；PyTorch 寫 `*.pt`，CUDA legacy 與 `cuda_native`/autograd 寫 `*.npz`。 |

## Current reliability contracts

- `train.init_seed` controls torch/flex model initialization, while CUDA legacy and CPU/NumPy autograd use their own seeded init paths.
- String booleans are parsed with strict semantics; `"false"` and `"0"` do not become true through Python `bool()`.
- Dotted CLI overrides may update list elements such as `model.layers.1.out_features=7`.
- `cuda_legacy` validation reports malformed numeric fields as validation errors before compiling an `ExperimentConfig`.
- CUDA legacy runtime cleanup frees weights and velocity buffers even when training raises before final evaluation.
