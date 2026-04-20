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
8. [08_autograd.md](08_autograd.md)：MiniCNN 自己的 CPU/NumPy `Tensor`、`Parameter`、`backward()`、layers、`SGD`/`Adam` 與 `train-autograd` 用法。

目前專案中的 CIFAR-10 dual-backend 訓練入口是 `minicnn train-dual`。同一份 config 可以用 `engine.backend=torch` 跑 PyTorch 路徑，或用 `engine.backend=cuda_legacy` 跑手寫 CUDA `.so` 路徑。

可直接修改的架構範本放在 `templates/`。MNIST 範本走 `train-flex` 並使用 `dataset.type: mnist`，可在第一次執行時下載 IDX gzip 檔；CIFAR-10 範本包含 PyTorch-only 與 CUDA legacy compatible 版本：

```bash
minicnn train-flex --config templates/mnist/lenet_like.yaml
minicnn train-flex --config templates/mnist/mlp.yaml
minicnn train-flex --config templates/cifar10/vgg_mini.yaml
minicnn train-dual --config templates/cifar10/vgg_mini_cuda.yaml engine.backend=cuda_legacy
```

MiniCNN 另有自己的 CPU/NumPy autograd stack，位於 `src/minicnn/nn/tensor.py`、`src/minicnn/ops/` 與 `src/minicnn/nn/layers.py`。它支援自訂可微分 op（`Function` API）、`grad_clip`、step-decay scheduler、random/cifar10/mnist 資料集，適合不依賴 torch 的 framework 測試與小型教學範例。詳情見 `docs/08_autograd.md`。

CIFAR-10 legacy trainer 已拆成明確模組。`src/minicnn/training/train_cuda.py` 只負責資料、epoch、validation、checkpoint、LR reduction、early stop 與 final test evaluation；CUDA batch 級的 conv forward、FC forward、fused loss/accuracy、FC update、conv backward/update 放在 `src/minicnn/training/cuda_batch.py`。Torch baseline 的 batch 準備、單步訓練與 epoch loop 在 `src/minicnn/training/train_torch_baseline.py` 中分成 `prepare_augmented_batch()`、`train_torch_batch()` 與 `run_torch_epoch()`。

兩個 legacy trainer 共用 `src/minicnn/training/legacy_data.py` 載入/normalize CIFAR-10，並共用 `src/minicnn/training/loop.py` 的 `RunningMetrics`、`LrState`、`FitState`、`EpochTimer`、plateau LR reduction 與 epoch summary formatter。CUDA 路徑仍使用 `conv_backward_precol`、`conv_update_fused` 與 `BatchWorkspace`；`USE_CUBLAS=1` 時 `gemm_forward` 與 conv weight gradient 走 cuBLAS 快速路徑，`USE_CUBLAS=0` 時走手寫 CUDA fallback。

如果要判斷改架構時是否需要改 backward kernel，請看
`docs/dual_backend_guide.md` 的 `When Architecture Changes Require Code Changes`
決策表。

MNIST 教學的最小版本是 `docs/train_mnist_so.py`；較乾淨的重構版本是 `docs/train_mnist_so_full_cnn_frame.py`，它把 CUDA orchestration 拆成 `ConvBlock`、`DenseLayer`、dataclass cache、shape helper 與獨立 `SgdOptimizer`。

若要同時編譯兩種 native backend：

```bash
minicnn build --legacy-make --variant both --check
```

先檢查本機環境：

```bash
minicnn info
minicnn doctor
```

然後用同一份 CIFAR-10 config 比較：

```bash
minicnn train-torch --config configs/dual_backend_cnn.yaml train.epochs=1
minicnn train-cuda --config configs/dual_backend_cnn.yaml train.epochs=1
minicnn train-autograd --config configs/autograd_tiny.yaml train.epochs=1

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade

minicnn compare --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32 train.batch_size=32

minicnn validate-config --config configs/dual_backend_cnn.yaml
minicnn compile --config configs/autograd_tiny.yaml
```

訓練產生的最佳模型檔固定寫入：

```text
src/minicnn/training/models/
```

PyTorch backend 會寫入 `*_best.pt`，CUDA legacy backend 會寫入 `*_best_model_split.npz`，CPU/NumPy autograd trainer 會寫入 `*_autograd_best.npz`。`artifacts/` 仍只保存 metrics 與 summary 等實驗紀錄。

`cuda_legacy` 的 `.so` 會 lazy-load：`minicnn --help`、`validate-dual-config`、`prepare-data`、torch backend、以及純 import 測試不會因為 native library 不存在而失敗。只有第一次真正呼叫 CUDA helper 時才會載入 `.so`。

目前 config/runtime 的穩健性規則：

- `train.init_seed` 控制模型初始化；比較 backend 時請固定這個值。
- CLI override 支援 list index，例如 `model.layers.1.out_features=7`。
- 布林欄位使用 strict parser，`"false"` 不會被 Python `bool()` 誤判為 true。
- 同一 process 內切換 `runtime.cuda_variant` 或 `runtime.cuda_so` 會重設 cached native library handle。
- `maxpool_backward_nchw_status` 是優先使用的 status-returning native API；舊的 void ABI 保留相容性。

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

2026-04-19 在 RTX 3050 Laptop GPU 上跑過以下 smoke tests。最新完整矩陣使用 `features/backend-smoke-matrix/run_smoke_matrix.py`，`128` train、`32` validation、batch size `32`、`1` epoch：

| Backend | Native variant | Train acc | Val acc | Test acc | Epoch time |
|---|---|---:|---:|---:|---:|
| `torch` | PyTorch CUDA | `11.72%` | `0.00%` | `10.36%` | `1.3s` |
| `cuda_legacy` | `cublas` | `7.03%` | `6.25%` | `12.97%` | `0.2s` |
| `cuda_legacy` | `handmade` | `7.03%` | `6.25%` | `12.97%` | `0.2s` |

較早的 `256` train、`64` validation 指令如下，若要重跑可直接使用：

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

從乾淨環境重現 CIFAR-10 實驗時，先執行：

```bash
minicnn build --legacy-make --check
minicnn prepare-data
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```
