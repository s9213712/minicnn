# Layout and Debug Guide

This guide covers the layout rules, memory sizing, and CUDA debugging workflow for the native `.so`.

## Layout Rules

The `.so` uses both NCHW and CNHW layouts.

| Data | Typical layout |
|---|---|
| Raw image input | NCHW `(N, C, H, W)` |
| `im2col_forward` input | NCHW |
| `gemm_forward` conv output | CNHW `(OUT_C, N, outH, outW)` |
| `maxpool_forward_store` input/output | CNHW |
| `dense_forward` input | row-major `(N, features)`, produced by converting CNHW pool output to NCHW then flattening |
| `conv_backward` input | NCHW for input; `grad_out` can reuse the CNHW conv buffer |
| `conv_backward_precol` col | row-major `(C*KH*KW, N*outH*outW)` from the same layer's `im2col_forward` |

Recommended habit:

```text
Before image or next conv:   NCHW
Conv raw / activation / maxpool:  CNHW
Before dense:  CNHW -> NCHW -> flatten
```

## Known Performance Limits

- The CUDA legacy training path still converts between NCHW and CNHW multiple times. This is a correctness-first conservative design; reducing conversion points is the next optimization step.
- The evaluation path uses `try/finally` to ensure GPU pointer release on exceptions but still allocates scratch buffers per batch. Add `EvalWorkspace` to reuse buffers if evaluation becomes a bottleneck.
- Host-to-device uploads during evaluation are synchronous; CUDA streams for prefetch/overlap have not been added.
- With `USE_CUBLAS=1`, convolution weight gradients use cuBLAS GEMM; the `USE_CUBLAS=0` handwritten fallback exists as a no-cuBLAS fallback, not as the fastest path.

## Common Errors

### Wrong buffer size

CUDA allocation uses bytes:

```python
ptr = lib.gpu_malloc(num_float32 * 4)
```

For a NumPy array:

```python
ptr = lib.gpu_malloc(arr.nbytes)
```

### `ctypes.argtypes` mismatch

A missing argument or wrong type in `argtypes` often causes crashes or wrong results. Cross-check against [guide_c_api.md](guide_c_api.md).

### CNHW/NCHW not converted

Most common mistake:

```text
conv raw output is CNHW
but the next im2col_forward expects NCHW
```

Insert the conversion:

```python
lib.cnhw_to_nchw(d_conv_raw, d_conv_nchw, N, C, H, W)
```

### Double free

Call `gpu_free` exactly once per GPU pointer. Putting the same pointer in multiple cache lists causes a double-free at batch end.

## CUDA Error Check

The project uses:

```cpp
CUDA_CHECK(cudaMalloc(...));
CUDA_KERNEL_CHECK();
```

Release build `CUDA_KERNEL_CHECK()` only checks launch errors. `make -C cpp debug` or CMake Debug build defines `MINICNN_DEBUG_SYNC`, making `CUDA_KERNEL_CHECK()` call `cudaDeviceSynchronize()` to locate async kernel errors.

On CUDA runtime API errors, the program prints:

```text
CUDA error at src/memory.cu:8: cudaMalloc(&ptr, size) failed: ...
```

## Minimum Validation Checklist

After modifying any `.cu` file:

```bash
make -C cpp
make -C cpp check
python3 -u /tmp/so_function_check.py
cuda-memcheck python3 -u /tmp/so_function_check.py
```

Check optimizer symbols:

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep -E 'apply_momentum_update|conv_update_fused|clip_inplace'
```

Check precol API:

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep conv_backward_precol
```

Check status-returning maxpool API:

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep maxpool_backward_nchw_status
```

Validate unified config mapping:

```bash
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
```

Full CIFAR-10 training:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

## Common Environment Problems

### `CUDA driver version is insufficient for CUDA runtime version`

Check with:

```bash
nvidia-smi
```

Usually caused by sandbox restrictions, WSL GPU permissions, or driver/runtime mismatch.

### `compute-sanitizer` unavailable

Some WSL/WDDM environments report debugger interface unsupported. Fall back to:

```bash
cuda-memcheck python3 -u your_script.py
```

### Training accuracy not improving

Diagnose incrementally:

1. Run FC baseline only; confirm loss decreases.
2. Add Conv + ReLU, without Pool.
3. Add Pool.
4. Check gradient scale at each step. `conv_backward` / `conv_backward_precol` accumulate into the passed `grad_out`; the CIFAR trainer applies batch mean to logits gradients before backward. CUDA batch call order is in `src/minicnn/training/cuda_batch.py` → `train_cuda_batch()`.
5. With Momentum SGD, the velocity buffer must not be reset per batch; it must persist from start to end of training.
6. If you changed the loss, verify that `softmax_xent_grad_loss_acc` produces a reasonable loss scalar, correct count, and `(probs - one_hot) / N`.

## Config Debug Notes

- Fix `train.init_seed` before comparing backends to avoid initialization variance.
- CLI override can modify list elements: `model.layers.1.out_features=7`.
- Boolean strings use a strict parser: `"false"`, `"0"`, `"no"` all parse as false.
- `cuda_legacy` runtime variant resets the native library cache when switching in the same process. If you suspect the wrong `.so` is loaded, print `MINICNN_CUDA_VARIANT`, `MINICNN_CUDA_SO`, and `resolve_library_path()`.

---

# Layout 與 Debug 指南（中文）

本文整理使用 `.so` 時最容易出錯的 layout、記憶體大小與 CUDA debug 流程。

## Layout 規則

這個 `.so` 同時存在 NCHW 與 CNHW。

| 資料 | 常用 layout |
|---|---|
| 原始影像 input | NCHW，`(N, C, H, W)` |
| `im2col_forward` input | NCHW |
| `gemm_forward` conv output | CNHW，`(OUT_C, N, outH, outW)` |
| `maxpool_forward_store` input/output | CNHW |
| `dense_forward` input | 一般 row-major `(N, features)` |
| `conv_backward_precol` col | row-major `(C*KH*KW, N*outH*outW)` |

建議習慣：

```text
影像或下一層 conv 前：NCHW
conv raw / activation / maxpool：CNHW
進 dense 前：CNHW -> NCHW -> flatten
```

## 已知效能限制

- CUDA legacy 訓練路徑仍在 NCHW 與 CNHW 間轉換多次，正確性優先的保守設計；下一步是減少轉換點。
- Evaluation path 保證例外時釋放 GPU pointer，但仍每個 batch 分配暫存 buffer。
- Evaluation 的 H2D upload 仍是同步流程。

## 常見錯誤

**buffer 大小**：CUDA allocation 使用 bytes，float32 = `num_elements * 4`。

**argtypes 不一致**：少一個參數或型別錯誤，容易 crash 或結果錯誤。對照 [guide_c_api.md](guide_c_api.md)。

**CNHW/NCHW 沒轉**：最典型錯誤，conv raw 是 CNHW，但下一層 `im2col_forward` 需要 NCHW，需先呼叫 `cnhw_to_nchw`。

**double free**：每個 GPU pointer 只呼叫一次 `gpu_free`。

## CUDA error check

`CUDA_KERNEL_CHECK()` 在 release build 只檢查 launch error；debug build 啟用 `MINICNN_DEBUG_SYNC` 後會呼叫 `cudaDeviceSynchronize()`。

## 最小驗證清單

```bash
make -C cpp
make -C cpp check
python3 -u /tmp/so_function_check.py
cuda-memcheck python3 -u /tmp/so_function_check.py
```

## 訓練準確率不升

1. 只跑 FC baseline，確認 loss 下降。
2. 加 Conv + ReLU，不加 Pool。
3. 再加 Pool。
4. 每步檢查 gradient scale。
5. Momentum SGD 時，velocity buffer 不能每個 batch 重設。
6. 修改 loss 後，確認 `softmax_xent_grad_loss_acc` 輸出合理的 loss scalar 與 `(probs - one_hot) / N`。

## Config debug 說明

- 固定 `train.init_seed` 再比較 backend。
- CLI override 可直接改 list 元素，例如 `model.layers.1.out_features=7`。
- 布林字串 strict parser：`"false"`、`"0"`、`"no"` 都解析成 false。
- `cuda_legacy` runtime variant 在同一 process 切換時會重設 native library cache。
