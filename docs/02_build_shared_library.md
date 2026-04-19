# 編譯 shared library

本文說明如何從 `cpp/src/*.cu` 編譯出 CUDA shared library。

## 基本編譯

在專案根目錄執行：

```bash
cd minicnn
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
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86 -DUSE_CUBLAS=$(USE_CUBLAS)
LDFLAGS = -shared -o $(OUTPUT) -Xlinker -rpath,$(CUDA_HOME)/lib64
```

`USE_CUBLAS=1` 是預設快速路徑，會額外 link `-lcublas`，讓 `gemm_forward` 與 `conv_backward` 的 weight gradient 使用 cuBLAS `cublasSgemm`。

若要保留純手寫 CUDA kernel、不連結 cuBLAS：

```bash
minicnn build --legacy-make --no-cublas --check
```

若要使用快速 cuBLAS backend：

```bash
minicnn build --legacy-make --check
```

直接使用 Makefile 時：

```bash
make -C cpp cublas
make -C cpp handmade
make -C cpp check-variants
```

## 選擇 `.so`

`cuda_legacy` 預設載入 `cpp/libminimal_cuda_cnn.so`。若要指定 variant：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

也可以用明確路徑覆蓋：

```bash
MINICNN_CUDA_SO=cpp/libminimal_cuda_cnn_cublas.so \
  minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

`.so` 使用 lazy-load。也就是說，`minicnn --help`、`minicnn prepare-data`、`validate-dual-config`、torch backend 訓練與一般 package import 不會因為 `.so` 尚未編譯而失敗。只有真正進入 CUDA helper 呼叫時，才會載入並 bind C API symbols。

建置並檢查必要匯出符號：

```bash
minicnn build --legacy-make --check
```

目前預設 Makefile 會編進 `.so` 的 `.cu` 檔包含：

```text
core.cu, gpu_monitor.cu, network.cu, dense_layer.cu, backward.cu,
memory.cu, loss_layer.cu, conv_backward.cu, optimizer.cu,
reorganize.cu, reorganize_backward.cu, maxpool_backward_nchw.cu,
leaky_relu.cu, layer_norm.cu, maxpool_store.cu,
maxpool_backward_use_idx.cu, layout_convert.cu
```

若新增 `.cu` 檔並希望匯出其中函式，需先加入 `cpp/Makefile` 的 `SRCS` 編譯清單。

## GPU 架構參數

如果你的 GPU 不是 Ampere/RTX 30 系列，可能需要修改 `-arch=sm_86`。

| GPU 架構 | `-arch` |
|---|---|
| GTX 10 系列 Pascal | `sm_61` |
| RTX 20 系列 Turing | `sm_75` |
| RTX 30 系列 Ampere | `sm_86` |
| RTX 40 系列 Ada | `sm_89` |

## 檢查匯出符號

```bash
nm -D --defined-only cpp/libminimal_cuda_cnn.so
```

目前常用匯出符號包含：

```text
gpu_malloc, gpu_free, gpu_memcpy_h2d, gpu_memcpy_d2h, gpu_memset
im2col_forward, gemm_forward, dense_forward
conv_backward, conv_backward_precol, dense_backward_full
maxpool_forward_store, maxpool_backward_use_idx
nchw_to_cnhw, cnhw_to_nchw
leaky_relu_forward, leaky_relu_backward
softmax_forward, softmax_cross_entropy, softmax_backward
softmax_xent_grad_loss_acc, count_correct
apply_sgd_update, apply_momentum_update, conv_update_fused, clip_inplace
```

若更新了 `optimizer.cu`，重編後可以只檢查 optimizer symbol：

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep -E 'apply_|conv_update_fused|clip_inplace'
```

預期至少看到：

```text
apply_sgd_update
apply_momentum_update
conv_update_fused
clip_inplace
```

## 清理與重編

```bash
make -C cpp clean
make -C cpp
```

## Windows DLL

Windows native CUDA backend 使用 CMake 與 Visual Studio generator：

```powershell
.\scripts\build_windows_native.ps1 -Variant both
```

詳細需求與手動 CMake 指令請看 [07_windows_build.md](07_windows_build.md)。

## CUDA memory check

```bash
cuda-memcheck python3 -u your_script.py
```

目前 `compute-sanitizer` 在部分 WSL/WDDM 環境可能會因 debugger interface 不支援而不能用；此環境下 `cuda-memcheck` 可正常檢查。
