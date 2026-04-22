# Building the Native Shared Library

This guide explains how to compile CUDA shared libraries from `cpp/src/*.cu`.

## Basic Build

From the project root:

```bash
minicnn build --legacy-make --check
```

On success this produces:

```text
cpp/libminimal_cuda_cnn.so
```

This is the default filename for backward compatibility. To produce both cuBLAS and handmade GEMM variants simultaneously, use the variant build below.

## Build Both Variants

```bash
minicnn build --legacy-make --variant both --check
```

On success:

```text
cpp/libminimal_cuda_cnn_cublas.so
cpp/libminimal_cuda_cnn_handmade.so
```

Build each variant separately:

```bash
minicnn build --legacy-make --variant cublas --check
minicnn build --legacy-make --variant handmade --check
```

## Default Makefile Settings

```makefile
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
USE_CUBLAS ?= 1
OUTPUT ?= libminimal_cuda_cnn.so
CUDA_ARCH ?= sm_86
CFLAGS = -O3 -Xcompiler -fPIC -arch=$(CUDA_ARCH) -DUSE_CUBLAS=$(USE_CUBLAS)
LDFLAGS = -shared -o $(OUTPUT) -Xlinker -rpath,$(CUDA_HOME)/lib64
```

`USE_CUBLAS=1` is the default fast path, which links `-lcublas` so that `gemm_forward` and the weight gradient in `conv_backward` use cuBLAS `cublasSgemm`.

To keep the handwritten CUDA kernels only:

```bash
minicnn build --legacy-make --no-cublas --check
```

Using the Makefile directly:

```bash
make -C cpp cublas
make -C cpp handmade
make -C cpp check-variants
```

Override the GPU architecture:

```bash
minicnn build --legacy-make --cuda-arch sm_75 --variant cublas --check
make -C cpp CUDA_ARCH=sm_75 cublas
```

## Runtime Variant Selection

Switch the native library at training time using the same config:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

`runtime.cuda_so=cpp/libminimal_cuda_cnn_cublas.so` sets the full path explicitly. MiniCNN lazy-loads the `.so` and clears the cached `ctypes` handle when switching native libraries in the same Python process.

`--check` verifies required exported symbols. `maxpool_backward_nchw_status` is a status-returning compatibility symbol that lets the Python wrapper convert native parameter errors into catchable exceptions.

Before training, use `minicnn validate-dual-config` and `minicnn healthcheck`.
They now emit JSON-friendly output or short user-facing failures instead of raw tracebacks.

## GPU Architecture

| GPU series | `-arch` |
|---|---|
| GTX 10 Pascal | `sm_61` |
| RTX 20 Turing | `sm_75` |
| RTX 30 Ampere | `sm_86` |
| RTX 40 Ada | `sm_89` |

CMake build accepts numeric form without the `sm_` prefix:

```bash
minicnn build --cuda-arch 75 --variant cublas --check
cmake -S cpp -B cpp/build-cmake -DCMAKE_CUDA_ARCHITECTURES=75
```

## Checking Exported Symbols

```bash
nm -D --defined-only cpp/libminimal_cuda_cnn.so
```

Common exported symbols include:

```text
gpu_malloc, gpu_free, gpu_memcpy_h2d, gpu_memcpy_d2h, gpu_memset
im2col_forward, gemm_forward, dense_forward
conv_backward, conv_backward_precol, dense_backward_full
maxpool_forward_store, maxpool_backward_use_idx
nchw_to_cnhw, cnhw_to_nchw
leaky_relu_forward, leaky_relu_backward
softmax_forward, softmax_backward
softmax_xent_grad_loss_acc, count_correct
apply_sgd_update, apply_momentum_update, conv_update_fused, clip_inplace
```

Check optimizer symbols only:

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep -E 'apply_|conv_update_fused|clip_inplace'
```

## Clean and Rebuild

```bash
make -C cpp clean
make -C cpp
```

## Windows DLL

See [guide_windows_build.md](guide_windows_build.md) for CMake-based Windows builds using the PowerShell helper script:

```powershell
.\scripts\build_windows_native.ps1 -Variant both
```

## CUDA Memory Check

```bash
cuda-memcheck python3 -u your_script.py
```

Note: `compute-sanitizer` may fail in some WSL/WDDM environments due to missing debugger interface support. `cuda-memcheck` works normally in these environments.

---

# 編譯 Native Shared Library（中文）

本文說明如何從 `cpp/src/*.cu` 編譯出 CUDA shared library。

## 基本編譯

在專案根目錄執行：

```bash
minicnn build --legacy-make --check
```

成功後會產生：

```text
cpp/libminimal_cuda_cnn.so
```

這是相容舊流程的預設檔名。若要同時保留 cuBLAS 與手寫 GEMM 兩種實作，使用下面的 variant build。

## 編譯兩種 `.so`

```bash
minicnn build --legacy-make --variant both --check
```

成功後會同時產生：

```text
cpp/libminimal_cuda_cnn_cublas.so
cpp/libminimal_cuda_cnn_handmade.so
```

也可以分開編：

```bash
minicnn build --legacy-make --variant cublas --check
minicnn build --legacy-make --variant handmade --check
```

## 預設 Makefile 設定

```makefile
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
USE_CUBLAS ?= 1
OUTPUT ?= libminimal_cuda_cnn.so
CUDA_ARCH ?= sm_86
CFLAGS = -O3 -Xcompiler -fPIC -arch=$(CUDA_ARCH) -DUSE_CUBLAS=$(USE_CUBLAS)
LDFLAGS = -shared -o $(OUTPUT) -Xlinker -rpath,$(CUDA_HOME)/lib64
```

`USE_CUBLAS=1` 是預設快速路徑，會額外 link `-lcublas`。若要使用純手寫 CUDA kernel：

```bash
minicnn build --legacy-make --no-cublas --check
```

直接使用 Makefile：

```bash
make -C cpp cublas
make -C cpp handmade
make -C cpp check-variants
```

GPU 架構可用 `--cuda-arch` 或 Makefile 變數覆蓋：

```bash
minicnn build --legacy-make --cuda-arch sm_75 --variant cublas --check
make -C cpp CUDA_ARCH=sm_75 cublas
```

## Runtime Variant 選擇

訓練時可用同一份 config 切換 native library：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

正式訓練前建議先跑 `minicnn validate-dual-config` 與 `minicnn healthcheck`。
這些命令現在都會輸出 JSON-friendly 結果，或以簡短使用者訊息失敗，不再直接吐 raw traceback。

## GPU 架構

| GPU 系列 | `-arch` |
|---|---|
| GTX 10 系列 Pascal | `sm_61` |
| RTX 20 系列 Turing | `sm_75` |
| RTX 30 系列 Ampere | `sm_86` |
| RTX 40 系列 Ada | `sm_89` |

## 檢查匯出符號

```bash
nm -D --defined-only cpp/libminimal_cuda_cnn.so
```

## 清理與重編

```bash
make -C cpp clean
make -C cpp
```

## Windows DLL

詳細需求與手動 CMake 指令請看 [guide_windows_build.md](guide_windows_build.md)。

## CUDA 記憶體檢查

```bash
cuda-memcheck python3 -u your_script.py
```

部分 WSL/WDDM 環境 `compute-sanitizer` 可能因 debugger interface 不支援而無法使用；此環境下 `cuda-memcheck` 可正常運作。
