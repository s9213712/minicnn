# Project Structure Reference

This document maps the responsibilities of files under `cpp/include`, `cpp/src`, and the Python package. The default training path calls native-library exports (`.so` on Linux, `.dll` on Windows) through the flat C ABI via `ctypes`. The C++ layer classes are a secondary API for C++ examples and experiments.

## Backend Role Map

- `src/minicnn/flex/` is the torch reference implementation
- `src/minicnn/cuda_native/` is the primary native backend direction
- `src/minicnn/autograd/`, `src/minicnn/nn/`, and `src/minicnn/ops/` form the CPU-side correctness oracle
- `src/minicnn/training/` and parts of `src/minicnn/core/` keep the historical `cuda_legacy` path running within its maintenance boundary

## Directory Layout

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

`data/cifar-10-batches-py/`, `cpp/*.so`, `cpp/*.dll`, `cpp/*.lib`,
`cpp/Release/*`, `artifacts/models/*`, `__pycache__/`, `.pytest_cache/`, and
other runtime artifacts are local files not tracked in Git.

Output-path policy:

- persistent run outputs belong under `artifacts/`
- ad-hoc generated files belong under `outputs/`
- root-level scratch directories such as `path-policy-artifacts/` are treated as unmanaged local output and should not be used for new tooling

## include

| File | Purpose |
|---|---|
| `cuda_check.h` | CUDA error-checking macros. `CUDA_CHECK(expr)` validates runtime API return values; release builds use `CUDA_KERNEL_CHECK()` for launch errors only; enabling `MINICNN_DEBUG_SYNC` in debug builds forces `cudaDeviceSynchronize()`. |
| `cublas_check.h` | cuBLAS error-checking macros, shared by `core.cu` and `conv_backward.cu`. |
| `tensor.h` | `CudaTensor` RAII wrapper for the secondary C++ API: manages GPU tensor memory and provides host/device copy. |
| `network.h` | Secondary C++ layer interface declaring `ConvLayer`, `ReLULayer`, `MaxPoolLayer`; forward outputs are represented as `std::unique_ptr<CudaTensor>`. |
| `dense_layer.h` | `DenseLayer` declaration for the secondary C++ API; forward output is managed as an RAII pointer. |

The primary Python/CLI training path uses the flat C ABI and `ctypes`. `network.h`, `dense_layer.h`, and `tensor.h` are reserved for C++ usage scenarios as described in [guide_cpp_linking.md](guide_cpp_linking.md), and are not the default training entry point.

## src

| File | Purpose |
|---|---|
| `memory.cu` | Exports `gpu_malloc`, `gpu_free`, `gpu_memcpy_h2d`, `gpu_memcpy_d2h`, `gpu_memset` for managing GPU memory from Python/C++. |
| `core.cu` | Basic forward kernels: `im2col_forward`, `gemm_forward`, `apply_relu`, `apply_maxpool`. With `USE_CUBLAS=1`, `gemm_forward` uses cuBLAS; with `USE_CUBLAS=0` it uses a handwritten GEMM kernel. |
| `cublas_context.cu` | Creates and reuses a single cuBLAS handle shared by forward GEMM and convolution backward. |
| `backward.cu` | ReLU backward and NCHW maxpool backward without saved indices. |
| `conv_backward.cu` | Convolution backward: with `USE_CUBLAS=1`, weight gradients use im2col + cuBLAS GEMM; with `USE_CUBLAS=0`, falls back to a handwritten CUDA kernel. Input gradients always use a direct CUDA kernel. Training uses `conv_backward_precol` to reuse the forward im2col buffer. |
| `dense_layer.cu` | Fully-connected layer forward/backward: `dense_forward`, `dense_backward_full`. |
| `loss_layer.cu` | `softmax_forward`, `softmax_backward`, `softmax_xent_grad_loss_acc`, `count_correct`, and `gemm_backward`. Training uses the fused softmax cross-entropy kernel to produce loss sum, accuracy count, probabilities, and logits gradient in one pass. |
| `optimizer.cu` | Optimizer kernels: `apply_sgd_update` (pure SGD); `apply_momentum_update` (Momentum SGD); `conv_update_fused` (GPU-side weight decay + gradient clipping + momentum); `clip_inplace` (in-place GPU gradient clipping). |
| `layout_convert.cu` | `nchw_to_cnhw` and `cnhw_to_nchw`. Some kernels produce CNHW output; training frequently needs to convert between layouts. |
| `reorganize.cu` / `reorganize_backward.cu` | Legacy layout reorganization API. New code should prefer the explicit NCHW/CNHW functions in `layout_convert.cu`. |
| `maxpool_store.cu` | MaxPool forward with index storage: `maxpool_forward_store`. |
| `maxpool_backward_use_idx.cu` | MaxPool backward using stored indices, paired with `maxpool_forward_store`. |
| `maxpool_backward_nchw.cu` | NCHW MaxPool backward. Retains the legacy `void maxpool_backward_nchw(...)` ABI and adds `int maxpool_backward_nchw_status(...)` so Python wrappers can convert native parameter errors into catchable exceptions. |
| `leaky_relu.cu` | LeakyReLU forward/backward, with both CNHW and NCHW named variants. |
| `layer_norm.cu` | LayerNorm forward/backward. |
| `network.cu` | C++ layer class forward implementations for C++ use. `ConvLayer` reuses the im2col cache; ReLU uses an out-of-place kernel to avoid D2D copies. |
| `gpu_monitor.cu` | `check_gpu_status()` prints `used_bytes,total_bytes` via `cudaMemGetInfo`; does not spawn a shell subprocess. |

## Python Package

| File | Purpose |
|---|---|
| `src/minicnn/cli.py` | `minicnn` CLI entrypoint with build, prepare-data, train-flex, train-dual, train-autograd, compare, validate-config, compile, and cuda_native commands. |
| `src/minicnn/autograd/` | `Tensor`, `Parameter`, `Function`, `Context`, `no_grad`, and `backward` compatibility namespace. |
| `src/minicnn/compiler/` | Lightweight MiniCNN IR, config tracer, and fusion/cleanup passes; `compile` command currently stops at IR summary without an independent lowering stage. |
| `src/minicnn/config/parsing.py` | CLI/config scalar parser, strict boolean parser, and dotted-override list-index write helper. |
| `src/minicnn/core/build.py` | Native CUDA shared library build/check wrapper supporting default, cublas, handmade, and both variants. |
| `src/minicnn/core/cuda_backend.py` | Lazy `ctypes` loader for the native CUDA library; does not load `.so`/`.dll` on non-CUDA command imports. `reset_library_cache()` clears the cached handle when switching native variants in the same process. |
| `src/minicnn/cuda_native/` | Primary native backend direction; public CLI surface is beta-grade, with GPU-first `gpu_native_auto`, strict `gpu_native` for supported real-CUDA subsets, and `reference_numpy` retained as explicit fallback/parity path. Training helpers are split into focused `gpu_training_*` modules for linear, pool, normalization, convolution, base depthwise, single-pointwise bridge, and two-pointwise activation bridge families. Lowering is likewise split into focused `gpu_lowering_*` modules for normalization, shape aliases, merge ops, activations, conv/pool ops, and registry assembly. |
| `src/minicnn/data/` | CIFAR-10 and MNIST preparation and data loading. |
| `src/minicnn/flex/` | PyTorch reference implementation: flexible config-driven model builder, registry, and trainer. |
| `src/minicnn/training/train_cuda.py` | Historical `cuda_legacy` CIFAR-10 orchestration: data, epoch, validation, checkpoint, LR reduction, early stop, final test evaluation. |
| `src/minicnn/training/cuda_batch.py` | CUDA batch-level forward/loss/backward/update steps; called by `train_cuda.py` to isolate kernel orchestration from training control flow. |
| `src/minicnn/unified/` | Shared config compiler mapping supported configs to the torch reference path, the historical `cuda_legacy` path, or the beta-grade `cuda_native` path. Runtime orchestration is now split into focused modules for context, diagnostics, training plans, gpu-native batch dispatch routing/family helpers, and the epoch loop. |

## Current Reliability Guarantees

- `train.init_seed` controls torch/flex model initialization; CUDA legacy and CPU/NumPy autograd use their own seeded init paths.
- String booleans are parsed strictly: `"false"` and `"0"` do not become true through Python `bool()`.
- Dotted CLI overrides may update list elements such as `model.layers.1.out_features=7`.
- Invalid config files and invalid dotted overrides now fail at the CLI boundary with short exit-code-2 messages instead of Python tracebacks.
- `cuda_legacy` validation reports malformed numeric fields as validation errors before compiling an `ExperimentConfig`.
- CUDA legacy runtime cleanup frees weights and velocity buffers even when training raises before final evaluation.
- `healthcheck`, `doctor`, and `smoke` return JSON-friendly payloads for shell tooling and agents.
- Torch paths now fail early when `train.device=cuda` is requested on a runtime without CUDA support.

---

# 專案結構說明（中文）

本文整理 `cpp/include`、`cpp/src` 與 Python package 中各檔案的責任。預設訓練流程主要透過 `extern "C"` 匯出的 C API 呼叫 native library（Linux 用 `.so`，Windows 用 `.dll`），C++ layer 類別是 secondary API，供 C++ 端範例與實驗直接使用。

## Backend 角色對照

- `src/minicnn/flex/` 是 torch reference implementation
- `src/minicnn/cuda_native/` 是主要 native backend 方向
- `src/minicnn/autograd/`、`src/minicnn/nn/`、`src/minicnn/ops/` 共同形成 CPU 側 correctness oracle
- `src/minicnn/training/` 與 `src/minicnn/core/` 的部分模組維持歷史 `cuda_legacy` 路徑在維護邊界內可用

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

`data/cifar-10-batches-py/`、`cpp/*.so`、`cpp/*.dll`、`cpp/*.lib`、
`cpp/Release/*`、`artifacts/models/*`、`__pycache__/`、`.pytest_cache/`
與其他 runtime artifacts 都是本機檔案，不屬於 Git 版本內容。

## include

| 檔案 | 作用 |
|---|---|
| `cuda_check.h` | CUDA 錯誤檢查工具。`CUDA_CHECK(expr)` 檢查 runtime API 回傳值；release 版 `CUDA_KERNEL_CHECK()` 只檢查 launch error，debug build 啟用 `MINICNN_DEBUG_SYNC` 後才同步 GPU。 |
| `cublas_check.h` | cuBLAS 錯誤檢查工具，集中供 `core.cu` 與 `conv_backward.cu` 共用。 |
| `tensor.h` | Secondary C++ API 的 `CudaTensor` RAII 包裝，管理 GPU tensor 記憶體，提供 host/device copy。 |
| `network.h` | Secondary C++ layer 介面與 `ConvLayer`、`ReLULayer`、`MaxPoolLayer` 宣告；forward output 以 `std::unique_ptr<CudaTensor>` 表示所有權。 |
| `dense_layer.h` | Secondary C++ API 的 `DenseLayer` 宣告，forward output 以 RAII pointer 管理。 |

主要 Python/CLI 訓練路徑使用 flat C ABI 和 `ctypes`。`network.h`、`dense_layer.h`、`tensor.h` 保留給 [guide_cpp_linking.md](guide_cpp_linking.md) 的 C++ 使用場景，不是預設訓練入口。

## src

| 檔案 | 作用 |
|---|---|
| `memory.cu` | 匯出 `gpu_malloc`、`gpu_free`、`gpu_memcpy_h2d`、`gpu_memcpy_d2h`、`gpu_memset`，供 Python/C++ 管理 GPU 記憶體。 |
| `core.cu` | 基礎 forward kernel：`im2col_forward`、`gemm_forward`、`apply_relu`、`apply_maxpool`。`USE_CUBLAS=1` 時 `gemm_forward` 使用 cuBLAS；`USE_CUBLAS=0` 時使用手寫 GEMM kernel。 |
| `cublas_context.cu` | 集中建立並重用單一 cuBLAS handle，供 forward GEMM 與 convolution backward 共用。 |
| `backward.cu` | ReLU backward 與不保存 index 的 NCHW maxpool backward。 |
| `conv_backward.cu` | 卷積層 backward：`USE_CUBLAS=1` 時 weight gradient 使用 im2col + cuBLAS GEMM；`USE_CUBLAS=0` 時保留手寫 CUDA fallback。訓練主流程使用 `conv_backward_precol` 重用 forward im2col buffer。 |
| `dense_layer.cu` | 全連接層 forward/backward：`dense_forward`、`dense_backward_full`。 |
| `loss_layer.cu` | `softmax_forward`、`softmax_backward`、`softmax_xent_grad_loss_acc`、`count_correct` 與 `gemm_backward`。 |
| `optimizer.cu` | Optimizer kernel：`apply_sgd_update`（純 SGD）；`apply_momentum_update`（Momentum SGD）；`conv_update_fused`（GPU 端合併 weight decay、gradient clipping、momentum update）；`clip_inplace`（GPU in-place clipping）。 |
| `layout_convert.cu` | `nchw_to_cnhw` 與 `cnhw_to_nchw`。 |
| `maxpool_store.cu` | 帶 max index 的 maxpool forward：`maxpool_forward_store`。 |
| `maxpool_backward_use_idx.cu` | 搭配 `maxpool_forward_store` 做 maxpool backward。 |
| `maxpool_backward_nchw.cu` | NCHW maxpool backward，含 status-returning `maxpool_backward_nchw_status`。 |
| `leaky_relu.cu` | LeakyReLU forward/backward，含 CNHW 與 NCHW 命名版本。 |
| `layer_norm.cu` | LayerNorm forward/backward。 |
| `network.cu` | C++ layer 類別的 forward 實作，供 C++ 端使用。 |
| `gpu_monitor.cu` | `check_gpu_status()`，透過 `cudaMemGetInfo` 印出記憶體使用量。 |

## Python package

| 檔案 | 作用 |
|---|---|
| `src/minicnn/cli.py` | `minicnn` CLI entrypoint，提供 build、prepare-data、train-flex、train-dual、train-autograd、compare、validate-config、compile、cuda_native 等命令。 |
| `src/minicnn/autograd/` | `Tensor`、`Parameter`、`Function`、`Context`、`no_grad` 與 `backward` compatibility namespace。 |
| `src/minicnn/compiler/` | 輕量 MiniCNN IR、config tracer 與 fusion/cleanup pass。 |
| `src/minicnn/core/build.py` | native CUDA shared library build/check wrapper。 |
| `src/minicnn/core/cuda_backend.py` | native CUDA library 的 lazy `ctypes` loader；非 CUDA 指令 import 時不會主動載入 `.so`/`.dll`，`reset_library_cache()` 供同一 process 切換 native variant 時清掉舊 handle。 |
| `src/minicnn/cuda_native/` | 主要 native backend 方向；已有公開 CLI 介面，現為 beta 級，預設 GPU-first `gpu_native_auto` 並保留 `reference_numpy` fallback。 |
| `src/minicnn/training/train_cuda.py` | legacy CUDA CIFAR-10 orchestration 入口。 |
| `src/minicnn/training/cuda_batch.py` | CUDA batch 級 forward/loss/backward/update 步驟。 |
| `src/minicnn/unified/` | shared config compiler，將支援的 config 映射到 torch reference 路徑、歷史 `cuda_legacy` 路徑或 beta 級 `cuda_native` 路徑。 |

## 目前可靠性邊界

- `train.init_seed` 控制 torch/flex 模型初始化；CUDA legacy 與 CPU/NumPy autograd 使用各自的 seeded init 路徑。
- 布林字串使用 strict parser；`"false"`、`"0"` 不會透過 Python `bool()` 成為 true。
- Dotted CLI override 可更新 list 元素，例如 `model.layers.1.out_features=7`。
- 無效 config 或 dotted override 現在會在 CLI 邊界以簡短訊息和 exit code 2 失敗，不再直接吐出 Python traceback。
- `cuda_legacy` validation 會把格式錯誤的數值欄位報告為 validation error，而不是在 `ExperimentConfig` compile 後才失敗。
- `healthcheck`、`doctor`、`smoke` 會回傳 JSON-friendly payload，方便 shell 工具與 agent 使用。
- torch 路徑若要求 `train.device=cuda`，但當前 runtime 不支援 CUDA，現在會提早失敗。

## CUDA Native maintenance map

The CUDA-native GPU training surface is being split incrementally without breaking public imports. `gpu_training.py` stays as the compatibility-facing module, while focused `gpu_training_*` modules now isolate result types, shared helpers, linear, pool, LayerNorm-family, BatchNorm/GroupNorm, conv-family, base depthwise helpers, and pointwise bridge-family helper code. Normalization-family lowering has also been extracted into `gpu_lowering_norm.py`, alongside the earlier registry/utility helper splits. See `docs/cuda_native_large_file_inventory.md` for the active large-file cleanup queue.
