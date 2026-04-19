# libminimal_cuda_cnn.so 使用教學索引

這組文件說明如何編譯與使用 `cpp/libminimal_cuda_cnn.so`，並用 MNIST 示範如何從 Python 或 C++ 呼叫 CUDA C API 做訓練與驗證。

建議閱讀順序：

1. [01_project_files.md](01_project_files.md)：`cpp/include` 與 `cpp/src` 各 `.h/.cu` 檔案用途。
2. [02_build_shared_library.md](02_build_shared_library.md)：如何編譯 `.so`、調整 GPU 架構、檢查匯出符號。
3. [03_c_api_reference.md](03_c_api_reference.md)：主要 C API、函式 prototype、forward/backward/update 介面。
4. [04_python_ctypes_mnist.md](04_python_ctypes_mnist.md)：Python `ctypes` 載入 `.so`，以及 MNIST CNN 訓練範例。
5. [05_cpp_linking.md](05_cpp_linking.md)：C++ 如何連結 `.so`，以及最小 inference 範例。
6. [06_layout_and_debug.md](06_layout_and_debug.md)：NCHW/CNHW layout 規則、常見錯誤、`cuda-memcheck` 驗證流程。
7. [07_windows_build.md](07_windows_build.md)：Windows `.dll` 編譯流程與 PowerShell helper。
8. [08_autograd.md](08_autograd.md)：MiniCNN 自己的 CPU/NumPy `Tensor`、`Parameter`、`backward()` 與輕量 `SGD` 用法。

目前專案中的 CIFAR-10 dual-backend 訓練入口是 `minicnn train-dual`。同一份 config 可以用 `engine.backend=torch` 跑 PyTorch 路徑，或用 `engine.backend=cuda_legacy` 跑手寫 CUDA `.so` 路徑。

MiniCNN 另有自己的小型 autograd core，位於 `src/minicnn/nn/tensor.py`。它適合不依賴 torch 的 framework 測試與小型教學範例；正式 torch backend 仍使用 PyTorch autograd，CUDA legacy backend 仍使用 CUDA/C++ backward kernels。

CIFAR-10 trainer 目前使用 `conv_backward_precol`、`conv_update_fused` 與 `BatchWorkspace`。`USE_CUBLAS=1` 時 `gemm_forward` 與 conv weight gradient 走 cuBLAS 快速路徑；`USE_CUBLAS=0` 時走手寫 CUDA fallback。MNIST 教學仍偏向最小可讀範例；若要追求速度，請參考 CIFAR trainer 的 workspace 重用與 precol backward 寫法。

若要同時編譯兩種 native backend：

```bash
minicnn build --legacy-make --variant both --check
```

然後用同一份 CIFAR-10 config 比較：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

訓練產生的最佳模型檔固定寫入：

```text
src/minicnn/training/models/
```

PyTorch backend 會寫入 `*_best.pt`，CUDA legacy backend 會寫入 `*_best_model_split.npz`。`artifacts/` 仍只保存 metrics 與 summary 等實驗紀錄。

`cuda_legacy` 的 `.so` 會 lazy-load：`minicnn --help`、`validate-dual-config`、`prepare-data`、torch backend、以及純 import 測試不會因為 native library 不存在而失敗。只有第一次真正呼叫 CUDA helper 時才會載入 `.so`。

Debug 時可直接用 config override 控制訓練參數：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas \
  train.epochs=1 train.batch_size=32 \
  dataset.num_samples=128 dataset.val_samples=32
```

legacy trainer 也支援常用環境變數覆蓋：

```bash
MINICNN_EPOCHS=1 MINICNN_BATCH=32 MINICNN_N_TRAIN=128 MINICNN_N_VAL=32 \
  minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

## Backend smoke comparison

2026-04-19 在 RTX 3050 Laptop GPU 上跑過以下 smoke tests：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=torch train.device=cuda \
  train.epochs=1 dataset.num_samples=256 dataset.val_samples=64 train.batch_size=64 \
  project.artifacts_root=/tmp/minicnn_backend_compare project.run_name=torch-cuda-smoke

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas \
  train.epochs=1 dataset.num_samples=256 dataset.val_samples=64 train.batch_size=64 \
  project.artifacts_root=/tmp/minicnn_backend_compare project.run_name=cuda-cublas-smoke

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade \
  train.epochs=1 dataset.num_samples=256 dataset.val_samples=64 train.batch_size=64 \
  project.artifacts_root=/tmp/minicnn_backend_compare project.run_name=cuda-handmade-smoke
```

結果：

| Backend | Native variant | Train acc | Val acc | Test acc | Epoch time |
|---|---|---:|---:|---:|---:|
| `torch` | PyTorch CUDA | `10.16%` | `12.50%` | 此較早 run 未記錄 | 此較早 run 未記錄 |
| `cuda_legacy` | `cublas` | `12.50%` | `20.31%` | `14.00%` | `0.1s` |
| `cuda_legacy` | `handmade` | `12.50%` | `20.31%` | `14.00%` | `0.3s` |

本次修改另以 `features/backend-smoke-matrix/run_smoke_matrix.py` 跑過更小的快速矩陣：`128` train、`32` validation、batch size `32`、`1` epoch。結果為 torch CUDA train_acc `11.72%` / val_acc `0.00%` / test_acc `10.36%` / epoch time `1.3s`，cuda_legacy cublas train_acc `7.03%` / val_acc `6.25%` / test_acc `12.97%` / epoch time `0.2s`，cuda_legacy handmade train_acc `7.03%` / val_acc `6.25%` / test_acc `12.97%` / epoch time `0.2s`。

從乾淨環境重現 CIFAR-10 實驗時，先執行：

```bash
minicnn build --legacy-make --check
minicnn prepare-data
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```
