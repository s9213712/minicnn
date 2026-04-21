# C API Reference

`libminimal_cuda_cnn.so` exports functions through `extern "C"`, making them accessible from Python `ctypes`, C/C++, or any FFI.

## GPU Memory

```c
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);
void gpu_memset(void* dst, int value, size_t size);
```

`size` is in bytes, not elements. A `float32` buffer of N elements requires `N * 4` bytes.

## Forward API

```c
void im2col_forward(float* input, float* col,
                    int N, int C, int H, int W,
                    int KH, int KW, int outH, int outW);

void gemm_forward(float* A, float* B, float* C,
                  int M, int N, int K);

void dense_forward(float* input, float* weights, float* bias, float* output,
                   int N, int in_f, int out_f);

void apply_relu(float* data, int size);
void leaky_relu_forward(float* data, float alpha, int size);

void maxpool_forward_store(float* output, float* input, int* max_idx,
                           int N, int C, int H, int W);

void softmax_forward(float* input, float* output, int N, int features);

void softmax_xent_grad_loss_acc(float* logits, int* labels,
                                float* probs, float* grad_logits,
                                float* loss_sum, int* correct_count,
                                int N, int features);

void count_correct(float* logits, int* labels, int* correct_count,
                   int N, int features);
```

Typical convolution forward sequence:

```text
NCHW input
-> im2col_forward
-> gemm_forward(weights, col, conv_raw)
-> activation
```

`gemm_forward` exposes a row-major API. With `USE_CUBLAS=1` it internally uses cuBLAS `cublasSgemm`; with `USE_CUBLAS=0` it falls back to a handwritten CUDA GEMM kernel. Conv output is typically CNHW: `(OUT_C, N, outH, outW)`.

## Backward API

```c
void dense_backward_full(float* d_out, float* input, float* weights,
                         float* d_input, float* d_weights, float* d_bias,
                         int N, int in_f, int out_f);

void conv_backward(float* grad_out, float* input, float* weights,
                   float* grad_weights, float* grad_input,
                   int N, int C, int H, int W,
                   int KH, int KW, int outH, int outW, int OUT_C);

void conv_backward_precol(float* grad_out, float* input, float* weights,
                          float* grad_weights, float* grad_input,
                          float* col,
                          int N, int C, int H, int W,
                          int KH, int KW, int outH, int outW, int OUT_C);

void leaky_relu_backward(float* data, float* grad, float alpha, int size);

void maxpool_backward_use_idx(float* grad_out, int* max_idx, float* grad_input,
                              int N, int C, int H, int W);

int maxpool_backward_nchw_status(float* grad_out, float* input, float* grad_input,
                                 int N, int C, int in_h, int in_w,
                                 int out_h, int out_w);

void maxpool_backward_nchw(float* grad_out, float* input, float* grad_input,
                           int N, int C, int in_h, int in_w,
                           int out_h, int out_w);

void softmax_backward(float* grad_out, float* probs, int N, int features);
```

`dense_backward_full` expects `d_out` to already contain the loss-reduction scaling. For example, if softmax cross-entropy uses batch mean, divide by N before passing `d_out`. The function does not divide by N itself.

`conv_backward` computes `grad_weights` (dL/dW) and `grad_input` (dL/dInput). With `USE_CUBLAS=1`, `grad_weights` uses im2col + cuBLAS GEMM. `conv_backward_precol` accepts an existing `col` buffer produced by a prior `im2col_forward` call, avoiding a redundant im2col pass.

`maxpool_backward_use_idx` is the safe training path, paired with stored `max_idx` from `maxpool_forward_store`. `maxpool_backward_nchw_status` returns `0` on success or a CUDA error code if `in_h != out_h * 2` or `in_w != out_w * 2`. The legacy `void maxpool_backward_nchw` is retained for ABI compatibility.

## Optimizer

```c
void apply_sgd_update(float* weights, float* grad, float lr, int size);

void apply_momentum_update(float* weights, float* grad, float* velocity,
                           float lr, float momentum, int size);

void conv_update_fused(float* weights, float* grad, float* velocity,
                       float lr, float momentum, float weight_decay,
                       float clip_val, float normalizer, int size);

void clip_inplace(float* values, float clip_val, int size);
```

`apply_sgd_update`: `weights[i] -= lr * grad[i]`

`apply_momentum_update`:
```text
velocity[i] = momentum * velocity[i] - lr * grad[i]
weights[i] += velocity[i]
```

`velocity` must be a GPU buffer of the same length as `weights`, zero-initialized before training and retained for the entire training run.

`conv_update_fused` performs on the GPU:
```text
g = grad[i] / normalizer + weight_decay * weights[i]
g = clip(g, -clip_val, clip_val)
velocity[i] = momentum * velocity[i] - lr * g
weights[i] += velocity[i]
```

Despite the name, `conv_update_fused` can also be used for FC weight/bias updates; set `weight_decay=0.0` for bias updates.

`clip_inplace` performs in-place element-wise clipping on a GPU buffer.

## Layout Conversion

```c
void nchw_to_cnhw(float* input, float* output, int N, int C, int H, int W);
void cnhw_to_nchw(float* input, float* output, int N, int C, int H, int W);
```

Layout summary:

```text
im2col_forward input:          NCHW
conv raw output:               CNHW  (OUT_C, N, outH, outW)
maxpool_forward_store I/O:     CNHW
dense_forward input:           row-major (N, features)
```

## Softmax Loss

`softmax_xent_grad_loss_acc` is the fused loss helper used by the CIFAR trainer. Python orchestration lives in `src/minicnn/training/cuda_batch.py` → `compute_loss_and_metrics()`. It computes:

```text
probs = softmax(logits)
grad_logits = (probs - one_hot(labels)) / N
loss_sum = sum(-log(probs[label]))
correct_count = number of argmax(logits) == label
```

`labels` are `int32` class IDs, not one-hot. Clear `loss_sum` and `correct_count` to zero before calling. The Python side only needs to download these two scalars.

`count_correct` is used during evaluation to compute batch correct count from logits on the GPU.

---

# C API 參考（中文）

`libminimal_cuda_cnn.so` 主要透過 `extern "C"` 匯出函式，適合 Python `ctypes`、C/C++ 或其他 FFI 呼叫。

## GPU 記憶體

```c
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);
void gpu_memset(void* dst, int value, size_t size);
```

`size` 是 bytes，不是元素數。`float32` buffer 大小通常是 `num_elements * 4`。

## Forward API

典型卷積 forward：

```text
NCHW input
-> im2col_forward
-> gemm_forward(weights, col, conv_raw)
-> activation
```

`gemm_forward` 以 row-major API 暴露。`USE_CUBLAS=1` 時內部使用 cuBLAS `cublasSgemm`；`USE_CUBLAS=0` 時使用手寫 CUDA GEMM kernel。Conv output 常用作 CNHW：`(OUT_C, N, outH, outW)`。

## Backward API

`dense_backward_full` 預期 `d_out` 已經包含 loss reduction 的縮放。

`conv_backward_precol` 與 `conv_backward` 行為相同，但多接收一個已存在的 `col` buffer，避免重新 im2col。

`maxpool_backward_use_idx` 是訓練主流程使用的安全路徑。`maxpool_backward_nchw_status` 是 status-returning 版本：成功回傳 `0`，若幾何不合則回傳 CUDA error code。

## Optimizer

`conv_update_fused` 在 GPU 端完成 weight decay + gradient clipping + Momentum update。Bias update 時將 `weight_decay` 設為 `0.0`。

## Layout 轉換

```text
im2col_forward input: NCHW
conv raw output: CNHW
maxpool_forward_store input/output: CNHW
dense_forward input: row-major (N, features)
```

## Softmax Loss

`softmax_xent_grad_loss_acc` 是目前 CIFAR trainer 使用的 fused loss helper。Python orchestration 位於 `src/minicnn/training/cuda_batch.py` 的 `compute_loss_and_metrics()`。`labels` 是 `int32` class id，不是 one-hot。呼叫前需將 `loss_sum` 與 `correct_count` 清為 0。
