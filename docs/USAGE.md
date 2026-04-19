# libminimal_cuda_cnn.so 使用教學索引

這組文件說明如何編譯與使用 `cpp/libminimal_cuda_cnn.so`，並用 MNIST 示範如何從 Python 或 C++ 呼叫 CUDA C API 做訓練與驗證。

建議閱讀順序：

1. [01_project_files.md](01_project_files.md)：`cpp/include` 與 `cpp/src` 各 `.h/.cu` 檔案用途。
2. [02_build_shared_library.md](02_build_shared_library.md)：如何編譯 `.so`、調整 GPU 架構、檢查匯出符號。
3. [03_c_api_reference.md](03_c_api_reference.md)：主要 C API、函式 prototype、forward/backward/update 介面。
4. [04_python_ctypes_mnist.md](04_python_ctypes_mnist.md)：Python `ctypes` 載入 `.so`，以及 MNIST CNN 訓練範例。
5. [05_cpp_linking.md](05_cpp_linking.md)：C++ 如何連結 `.so`，以及最小 inference 範例。
6. [06_layout_and_debug.md](06_layout_and_debug.md)：NCHW/CNHW layout 規則、常見錯誤、`cuda-memcheck` 驗證流程。

目前專案中的 CIFAR-10 dual-backend 訓練入口是 `minicnn train-dual`。同一份 config 可以用 `engine.backend=torch` 跑 PyTorch 路徑，或用 `engine.backend=cuda_legacy` 跑手寫 CUDA `.so` 路徑。

CIFAR-10 trainer 目前使用 `conv_backward_precol`、`conv_update_fused` 與 `BatchWorkspace`。`USE_CUBLAS=1` 時 `gemm_forward` 與 conv weight gradient 走 cuBLAS 快速路徑；`USE_CUBLAS=0` 時走手寫 CUDA fallback。MNIST 教學仍偏向最小可讀範例；若要追求速度，請參考 CIFAR trainer 的 workspace 重用與 precol backward 寫法。

從乾淨環境重現 CIFAR-10 實驗時，先執行：

```bash
minicnn build --legacy-make --check
minicnn prepare-data
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```
