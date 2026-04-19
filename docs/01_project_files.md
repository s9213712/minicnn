# 專案檔案說明

本文整理 `cpp/include` 與 `cpp/src` 中各檔案的責任。預設訓練流程主要透過 `extern "C"` 匯出的 C API 呼叫 `.so`，C++ layer 類別則偏向 C++ 端直接使用。

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
│   ├── data/
│   ├── flex/
│   ├── framework/
│   ├── models/
│   ├── nn/
│   ├── ops/
│   ├── optim/
│   ├── runtime/
│   ├── training/
│   │   └── models/
│   └── unified/
├── configs/
├── tests/
└── docs/
```

`data/cifar-10-batches-py/`、`cpp/*.so`、`src/minicnn/training/models/*`、`__pycache__/`、`.pytest_cache/` 與 runtime artifacts 都是本機檔案，不屬於 Git 版本內容。

## include

| 檔案 | 作用 |
|---|---|
| `cuda_check.h` | CUDA 錯誤檢查工具。`CUDA_CHECK(expr)` 檢查 runtime API 回傳值；`CUDA_KERNEL_CHECK()` 檢查 kernel launch error 並同步 GPU。 |
| `tensor.h` | `CudaTensor` C++ RAII 包裝，管理 GPU tensor 記憶體，提供 host/device copy。 |
| `network.h` | C++ layer 介面與 `ConvLayer`、`ReLULayer`、`MaxPoolLayer` 宣告。 |
| `dense_layer.h` | C++ `DenseLayer` 宣告。 |

## src

| 檔案 | 作用 |
|---|---|
| `memory.cu` | 匯出 `gpu_malloc`、`gpu_free`、`gpu_memcpy_h2d`、`gpu_memcpy_d2h`、`gpu_memset`，供 Python/C++ 管理 GPU 記憶體。 |
| `core.cu` | 基礎 forward kernel：`im2col_forward`、`gemm_forward`、`apply_relu`、`apply_maxpool`。`USE_CUBLAS=1` 時 `gemm_forward` 使用 cuBLAS；`USE_CUBLAS=0` 時使用手寫 GEMM kernel。 |
| `cublas_context.cu` | 集中建立並重用單一 cuBLAS handle，供 forward GEMM 與 convolution backward 共用。 |
| `backward.cu` | ReLU backward 與不保存 index 的 NCHW maxpool backward。 |
| `conv_backward.cu` | 卷積層 backward：`USE_CUBLAS=1` 時 weight gradient 使用 im2col + cuBLAS GEMM；`USE_CUBLAS=0` 時保留手寫 CUDA fallback。input gradient 仍使用直接 CUDA kernel。訓練主流程使用 `conv_backward_precol` 重用 forward im2col buffer。 |
| `dense_layer.cu` | 全連接層 forward/backward：`dense_forward`、`dense_backward_full`。 |
| `loss_layer.cu` | `softmax_forward`、`softmax_backward`、`softmax_xent_grad_loss_acc`、`count_correct`，另含 `im2col_backward`、`gemm_backward`。 |
| `optimizer.cu` | Optimizer kernel。`apply_sgd_update` 執行純 SGD；`apply_momentum_update` 執行 Momentum SGD；`conv_update_fused` 在 GPU 端合併 weight decay、gradient clipping、momentum update；`clip_inplace` 做 GPU in-place gradient clipping。 |
| `layout_convert.cu` | `nchw_to_cnhw` 與 `cnhw_to_nchw`。部分 kernel 輸出以 CNHW 儲存，訓練時常需要轉換。 |
| `reorganize.cu` / `reorganize_backward.cu` | 舊版 layout 重排 API。新程式建議優先用 `layout_convert.cu` 的明確 NCHW/CNHW 函式。 |
| `maxpool_store.cu` | 帶 max index 的 maxpool forward：`maxpool_forward_store`。 |
| `maxpool_backward_use_idx.cu` | 搭配 `maxpool_forward_store` 做 maxpool backward。 |
| `maxpool_backward_nchw.cu` | NCHW maxpool backward 版本。 |
| `leaky_relu.cu` | LeakyReLU forward/backward，含 CNHW 與 NCHW 命名版本。 |
| `layer_norm.cu` | LayerNorm forward/backward。 |
| `network.cu` | C++ layer 類別的 forward 實作，供 C++ 端使用。 |
| `gpu_monitor.cu` | `check_gpu_status()`，呼叫 `nvidia-smi` 印出 GPU 使用率。 |

## Python package

| 檔案 | 作用 |
|---|---|
| `src/minicnn/cli.py` | `minicnn` CLI entrypoint，提供 build、prepare-data、train-flex、train-dual、train-autograd、compare、validate-config、compile 等命令。 |
| `src/minicnn/autograd/` | `Tensor`、`Parameter`、`Function`、`Context`、`no_grad` 與 `backward` compatibility namespace。 |
| `src/minicnn/compiler/` | 輕量 MiniCNN IR、config tracer、fusion 標記 pass、scheduler 與明確的 lowering 邊界。 |
| `src/minicnn/core/build.py` | native CUDA shared library build/check wrapper，支援 default、cublas、handmade、both variants。 |
| `src/minicnn/core/cuda_backend.py` | native CUDA library 的 lazy `ctypes` loader；非 CUDA 指令 import 時不會載入 `.so`。 |
| `src/minicnn/core/fused_ops.py` | Conv2d + BatchNorm2d + ReLU fusion 語意的 NumPy reference helper。 |
| `src/minicnn/data/` | CIFAR-10 準備與資料載入。 |
| `src/minicnn/flex/` | PyTorch flexible config-driven model builder、registry、trainer。 |
| `src/minicnn/models/` | CPU/NumPy MiniCNN model registry、shape inference、config builder 與 graph helper。 |
| `src/minicnn/nn/` | MiniCNN framework layer，包含 `Module`、`Sequential`、`Tensor`、`Parameter` 與 CPU/NumPy autograd functions。 |
| `src/minicnn/nn/tensor.py` | reverse-mode autograd engine；支援 arithmetic、broadcasting、matmul、reductions、reshape、ReLU、`log_softmax`、`cross_entropy` 與 `Tensor.backward()`。 |
| `src/minicnn/nn/layers.py` | CPU/NumPy MiniCNN layers：`Linear`、`Conv2d`、`MaxPool2d`、`BatchNorm2d`、`Flatten`、`ReLU`、`ResidualBlock`。 |
| `src/minicnn/ops/` | MiniCNN layers 使用的 differentiable NumPy ops。 |
| `src/minicnn/optim/` | 輕量 optimizer 介面；`SGD` 與 `Adam` 可在不依賴 torch 的情況下更新 MiniCNN `Parameter`。 |
| `src/minicnn/runtime/` | 小型 graph executor、backend protocol、tensor memory pool、profiler utilities。 |
| `src/minicnn/unified/` | shared config compiler，將支援的 config 映射到 `torch` 或 `cuda_legacy` backend。 |
| `src/minicnn/training/train_cuda.py` | legacy CUDA CIFAR-10 training loop 入口。 |
| `src/minicnn/training/train_autograd.py` | random-data CPU/NumPy autograd training loop，輸出 `*_autograd_best.npz`。 |
| `src/minicnn/training/cuda_ops.py` | legacy loop 使用的 CUDA copy、layout、forward wrapper。 |
| `src/minicnn/training/cuda_workspace.py` | batch GPU workspace，重用每 batch buffer 並保護 double-free。 |
| `src/minicnn/training/evaluation.py` | CUDA evaluation forward path 與 accuracy helper。 |
| `src/minicnn/training/checkpoints.py` | CUDA checkpoint save/reload 與 GPU pointer cleanup。 |
| `src/minicnn/training/train_torch_baseline.py` | 對齊 CUDA update 規則的 PyTorch baseline。 |
| `src/minicnn/training/models/` | 最佳模型固定輸出位置；PyTorch 寫 `*.pt`，CUDA legacy 寫 `*.npz`。 |
