# C++ Linking Guide

This guide explains how to link the MiniCNN native library from a C++ program.

- Linux: `cpp/libminimal_cuda_cnn.so`
- Windows: `cpp\\Release\\minimal_cuda_cnn.lib` with the matching `.dll`

The default training path uses Python `ctypes` against the flat C ABI. `network.h`, `dense_layer.h`, and `tensor.h` provide a secondary C++ API for C++ examples and experiments. For CIFAR-10 training, prefer `minicnn train-dual`.

## Basic Approach

Because the native library exports a C ABI, C++ programs can declare the same prototypes directly:

```cpp
extern "C" {
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);

void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output,
                   int N, int in_f, int out_f);

void conv_backward_precol(float* grad_out, float* input, float* weights,
                          float* grad_weights, float* grad_input,
                          float* col,
                          int N, int C, int H, int W,
                          int KH, int KW, int outH, int outW, int OUT_C);

void conv_update_fused(float* weights, float* grad, float* velocity,
                       float lr, float momentum, float weight_decay,
                       float clip_val, float normalizer, int size);

int maxpool_backward_nchw_status(float* grad_out, float* input, float* grad_input,
                                 int N, int C, int in_h, int in_w,
                                 int out_h, int out_w);
}
```

When calling native helpers that expose a status-returning variant, check the returned CUDA error code before continuing. The older void ABI is still present for compatibility, but the status form is easier to integrate into host-side error handling.

## Minimal Inference Example

Create `examples/mnist_infer_demo.cpp`:

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);

void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output,
                   int N, int in_f, int out_f);
}

int main() {
    constexpr int N = 2;
    constexpr int IN = 28 * 28;
    constexpr int OUT = 10;

    std::vector<float> x(N * IN, 0.0f);
    std::vector<float> w(OUT * IN, 0.001f);
    std::vector<float> b(OUT, 0.0f);
    std::vector<float> y(N * OUT, 0.0f);

    float* d_x = static_cast<float*>(gpu_malloc(x.size() * sizeof(float)));
    float* d_w = static_cast<float*>(gpu_malloc(w.size() * sizeof(float)));
    float* d_b = static_cast<float*>(gpu_malloc(b.size() * sizeof(float)));
    float* d_y = static_cast<float*>(gpu_malloc(y.size() * sizeof(float)));

    gpu_memcpy_h2d(d_x, x.data(), x.size() * sizeof(float));
    gpu_memcpy_h2d(d_w, w.data(), w.size() * sizeof(float));
    gpu_memcpy_h2d(d_b, b.data(), b.size() * sizeof(float));

    dense_forward(d_x, d_w, d_b, d_y, N, IN, OUT);
    gpu_memcpy_d2h(y.data(), d_y, y.size() * sizeof(float));

    for (int n = 0; n < N; ++n) {
        int argmax = 0;
        for (int j = 1; j < OUT; ++j) {
            if (y[n * OUT + j] > y[n * OUT + argmax]) argmax = j;
        }
        std::printf("sample %d pred=%d logit=%f\n", n, argmax, y[n * OUT + argmax]);
    }

    gpu_free(d_x); gpu_free(d_w); gpu_free(d_b); gpu_free(d_y);
    return 0;
}
```

Compile:

```bash
cd minicnn
g++ examples/mnist_infer_demo.cpp \
  -Lcpp -lminimal_cuda_cnn \
  -Wl,-rpath,'$ORIGIN/../cpp' \
  -o examples/mnist_infer_demo
```

Run:

```bash
./examples/mnist_infer_demo
```

Windows note:

- build the DLL/import-library pair with `scripts/build_windows_native.ps1`
- link against the generated `.lib`
- keep the matching `.dll` beside the executable or on `PATH`
- see [guide_windows_build.md](guide_windows_build.md) for the validated build flow

## Full C++ Training Flow

For a complete C++ training loop, the sequence mirrors Python:

1. Prepare NCHW float32 input, e.g. MNIST `(N, 1, 28, 28)`.
2. Allocate input, weights, bias, and intermediate buffers with `gpu_malloc`. Fixed-shape batch buffers should be allocated once before training and reused across batches.
3. Forward: `im2col_forward → gemm_forward → activation → maxpool_forward_store → layout convert → dense_forward`.
4. Loss: use `softmax_xent_grad_loss_acc` to produce logits gradient, loss sum, and correct count on the GPU in one pass.
5. Backward: `dense_backward_full → layout convert → maxpool_backward_use_idx → activation backward → conv_backward_precol` (use `conv_backward` if no saved col buffer).
6. Update: use `conv_update_fused` to keep weight decay, gradient clipping, and momentum on the GPU. Each trainable buffer requires a velocity buffer of the same length, zero-initialized and retained for the full run.
7. After training, release workspace, weights, and velocities. Do not free fixed-shape scratch buffers per batch.

The Python CIFAR-10 CUDA orchestration is in `src/minicnn/training/cuda_batch.py`. Compare `train_cuda_batch()`, `forward_convs()`, `backward_convs()`, and `update_convs()` to understand the C API call order.

Momentum update C ABI:

```cpp
extern "C" {
void apply_momentum_update(float* weights, float* grad, float* velocity,
                           float lr, float momentum, int size);

void conv_update_fused(float* weights, float* grad, float* velocity,
                       float lr, float momentum, float weight_decay,
                       float clip_val, float normalizer, int size);
}
```

Update formulas:

```text
velocity[i] = momentum * velocity[i] - lr * grad[i]
weights[i] += velocity[i]
```

`conv_update_fused`:
```text
g = grad[i] / normalizer + weight_decay * weights[i]
g = clip(g, -clip_val, clip_val)
velocity[i] = momentum * velocity[i] - lr * g
weights[i] += velocity[i]
```

Set `weight_decay=0.0` for bias updates.

## Runtime Library Path

On Linux, the most common linking problem is the runtime failing to find the `.so`. Three options:

```bash
# Option 1: embed rpath at compile time
-Wl,-rpath,'$ORIGIN/../cpp'

# Option 2: set LD_LIBRARY_PATH at runtime
export LD_LIBRARY_PATH="$PWD/cpp:${LD_LIBRARY_PATH:-}"

# Option 3: install the .so to a standard linker search path
```

---

# C++ 連結與使用 Native Library（中文）

本文說明如何從 C++ 程式連結 MiniCNN 的 native library。

- Linux：`cpp/libminimal_cuda_cnn.so`
- Windows：`cpp\\Release\\minimal_cuda_cnn.lib`，搭配對應 `.dll`

預設訓練流程走 Python `ctypes` 呼叫 flat C ABI。`network.h`、`dense_layer.h`、`tensor.h` 提供的是 secondary C++ API，適合 C++ 範例與實驗；若只是要訓練 CIFAR-10，優先使用 `minicnn train-dual`。

## 基本方式

因為 native library 主要匯出 C ABI，C++ 程式可直接宣告相同 prototype（參考上方 `extern "C"` 區塊）。

有 status-returning 版本時，優先使用它（如 `maxpool_backward_nchw_status`），方便在 host 端做錯誤處理。

## 最小 inference 範例

參考上方 `examples/mnist_infer_demo.cpp`。

編譯：

```bash
cd minicnn
g++ examples/mnist_infer_demo.cpp \
  -Lcpp -lminimal_cuda_cnn \
  -Wl,-rpath,'$ORIGIN/../cpp' \
  -o examples/mnist_infer_demo
```

Windows 補充：

- 先用 `scripts/build_windows_native.ps1` 產出 `.dll` / `.lib`
- 連結時使用對應 `.lib`
- 執行時把配對的 `.dll` 放在可搜尋的位置
- 具體建置流程請看 [guide_windows_build.md](guide_windows_build.md)

## C++ 完整訓練流程

流程和 Python 相同：

1. 準備 NCHW float32 input。
2. 用 `gpu_malloc` 配置 input、weights、bias、intermediate buffer；固定 shape 的 batch buffer 建議訓練前配置一次，跨 batch 重用。
3. Forward：`im2col_forward → gemm_forward → activation → maxpool_forward_store → layout convert → dense_forward`。
4. Loss：使用 `softmax_xent_grad_loss_acc` 在 GPU 端產生 logits gradient、loss sum 與 correct count。
5. Backward：`dense_backward_full → layout convert → maxpool_backward_use_idx → activation backward → conv_backward_precol`。
6. Update：使用 `conv_update_fused` 在 GPU 端合併 weight decay、gradient clipping、Momentum update。
7. 訓練結束後釋放 workspace、weights、velocity；不要每個 batch 反覆釋放固定 shape 的暫存 buffer。

Python 端的 CIFAR-10 CUDA orchestration 集中在 `src/minicnn/training/cuda_batch.py`，可對照 C API 呼叫順序。

## 注意事項

在 Linux 上，連結時最常見問題是 runtime 找不到 `.so`。三種解法：

```bash
# 方式 1：編譯時寫 rpath
-Wl,-rpath,'$ORIGIN/../cpp'

# 方式 2：執行前設定 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$PWD/cpp:${LD_LIBRARY_PATH:-}"

# 方式 3：把 .so 放到系統 linker 搜尋路徑
```
