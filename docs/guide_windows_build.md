# Windows Native Build

> Status: manually validated on a real Windows machine.
>
> This path is still not covered by CI, but the repository now includes a
> tested manual workflow for building the native CUDA backend as Windows
> `.dll` files.

This page documents the validated manual Windows build path for the native CUDA
backend.

## Requirements

- Windows 10/11
- NVIDIA display driver with CUDA support
- Visual Studio 2019 or 2022 with the C++ workload
- CMake 3.20 or newer
- CUDA Toolkit installed on Windows
- PowerShell

## Validated Path

The branch-local validation record for this repo used:

- `Generator = Visual Studio 16 2019`
- `Platform = x64`
- `Config = Release`
- `CMAKE_CUDA_ARCHITECTURES = 75`

The PowerShell helper keeps `-Generator`, `-Platform`, and `-CudaArch`
overrideable. The project does not hardcode a Visual Studio generator inside
`CMakeLists.txt`.

## Source Encoding Note

Do not rely on `/utf-8` to make native headers with non-ASCII comments compile
on Windows. The safer repo-side fix here is to keep the handwritten CUDA/C++
headers ASCII-only, so the validated Windows path does not depend on a
UTF-16-vs-UTF-8 source encoding choice.

## Preflight Checks

Run these first in PowerShell before configuring CMake:

```powershell
nvidia-smi
nvcc --version
cmake --version
dir "C:\Program Files (x86)\Windows Kits\10\Include"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

If `nvidia-smi` fails, fix the driver path before debugging CMake or CUDA.

## Build Both Native Variants

Run from the repository root in PowerShell. This matches the validated branch
local path:

```powershell
.\scripts\build_windows_native.ps1 -Variant both -Clean
```

The helper prints the resolved repo/build paths, the exact `cmake` arguments,
and the discovered DLL locations after each successful build.

For other GPU generations, override `-CudaArch` explicitly:

```powershell
.\scripts\build_windows_native.ps1 -Variant both -Clean -CudaArch 89
```

Expected outputs:

```text
cpp\Release\minimal_cuda_cnn_cublas.dll
cpp\Release\minimal_cuda_cnn_cublas.lib
cpp\Release\minimal_cuda_cnn_handmade.dll
cpp\Release\minimal_cuda_cnn_handmade.lib
```

The cuBLAS variant compiles with `USE_CUBLAS=ON`. The handmade variant compiles with `USE_CUBLAS=OFF`.

## Build One Variant

```powershell
.\scripts\build_windows_native.ps1 -Variant cublas
.\scripts\build_windows_native.ps1 -Variant handmade
.\scripts\build_windows_native.ps1 -Variant default
```

## GPU Architecture

| GPU series | `CMAKE_CUDA_ARCHITECTURES` |
|---|---:|
| RTX 20 Turing | `75` |
| RTX 30 Ampere | `86` |
| RTX 40 Ada | `89` |
| RTX 50 Blackwell | `120` |

## Manual CMake Command

```powershell
$CudaArch = "75"
cmake -S cpp -B cpp\build-windows-cublas -G "Visual Studio 16 2019" -A x64 `
  -DUSE_CUBLAS=ON `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_cublas `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-cublas --config Release --parallel
```

For the handmade variant:

```powershell
$CudaArch = "75"
cmake -S cpp -B cpp\build-windows-handmade -G "Visual Studio 16 2019" -A x64 `
  -DUSE_CUBLAS=OFF `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_handmade `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-handmade --config Release --parallel
```

If you switch generator, platform, SDK, or CUDA architecture, clear the old
build directory first. `.\scripts\build_windows_native.ps1 -Clean` does this
for you.

## Python Loading Note

The Python loader maps native variants to `.dll` names on Windows and adds `cpp/` plus `%CUDA_PATH%\bin` to the DLL search path before `ctypes.CDLL()`.

The C API export surface is kept the same, so the intended Windows loader mapping is:

```text
runtime.cuda_variant=cublas   -> cpp\minimal_cuda_cnn_cublas.dll
runtime.cuda_variant=handmade -> cpp\minimal_cuda_cnn_handmade.dll
```

The Python loader also resets its cached native handle when a process switches variants, so Windows smoke tests should verify both DLLs in the same order used on Linux:

```powershell
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy runtime.cuda_variant=cublas train.epochs=1
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy runtime.cuda_variant=handmade train.epochs=1
```

As on Linux, run `minicnn validate-dual-config` and `minicnn healthcheck`
first. They now return JSON-friendly output or short user-facing failures.

Both DLL variants should export `maxpool_backward_nchw_status` in addition to the legacy void `maxpool_backward_nchw` symbol.

Treat these as the manual success criteria:

- the expected `.dll` and `.lib` files exist under `cpp\Release\`
- both variants export `maxpool_backward_nchw_status`
- `minicnn validate-dual-config` and `minicnn healthcheck` pass
- both `runtime.cuda_variant=cublas` and `runtime.cuda_variant=handmade` load
  in separate smoke runs

## DLL Usage Code

The repo now includes a direct Python ctypes smoke example for `.dll` loading:

```powershell
python -u examples\mnist_ctypes\check_native_library.py --variant handmade
python -u examples\mnist_ctypes\check_native_library.py --path cpp\minimal_cuda_cnn_cublas.dll
```

For a fuller training-oriented example, use:

```powershell
python -u examples\mnist_ctypes\train_mnist_so_full_cnn_frame.py --download
```

The smoke script goes through MiniCNN's own DLL resolver and binding layer,
checks the required symbols, then performs a GPU upload/download round-trip.

---

# Windows Native Build（中文）

> 狀態：已在實際 Windows 機器上做過手動驗證。
>
> 這條路徑目前仍沒有 CI 覆蓋，但 repo 內已整理出一條實測過的
> Windows `.dll` 手動建置流程。

本文說明 Windows 平台 native CUDA backend 的手動驗證建置流程。

## 需求

- Windows 10/11
- 支援 CUDA 的 NVIDIA 顯示驅動
- Visual Studio 2019 或 2022（含 C++ 工作負載）
- CMake 3.20 以上
- Windows 端安裝的 CUDA Toolkit
- PowerShell

## 已驗證路徑

這次 repo 內保留的 Windows 建置紀錄使用了：

- `Generator = Visual Studio 16 2019`
- `Platform = x64`
- `Config = Release`
- `CMAKE_CUDA_ARCHITECTURES = 75`

PowerShell helper 仍保留 `-Generator`、`-Platform`、`-CudaArch` 可覆寫。
專案本身不會在 `CMakeLists.txt` 內寫死 Visual Studio generator。

## 原始碼編碼注意事項

不要依賴 `/utf-8` 來讓含非 ASCII 註解的 native header 在 Windows 上通過編譯。
這次 repo 端採用的穩定做法，是把手寫 CUDA/C++ header 維持為 ASCII-only，
避免整條 Windows build 路徑卡在 UTF-16 和 UTF-8 的差異上。

## 建置前檢查

在 PowerShell 先執行：

```powershell
nvidia-smi
nvcc --version
cmake --version
dir "C:\Program Files (x86)\Windows Kits\10\Include"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

如果 `nvidia-smi` 先失敗，優先修顯示卡驅動與 CUDA 環境，不要直接往
CMake 設定排查。

## 編譯兩種 Native Variant

在 PowerShell 專案根目錄執行。下面這條就是這次驗證路徑對齊的用法：

```powershell
.\scripts\build_windows_native.ps1 -Variant both -Clean
```

helper 會印出解析後的 repo/build 路徑、完整 `cmake` 參數，以及成功後找到的
DLL 實際路徑。

若使用其他 GPU 世代，請明確覆寫 `-CudaArch`：

```powershell
.\scripts\build_windows_native.ps1 -Variant both -Clean -CudaArch 89
```

預期輸出：

```text
cpp\Release\minimal_cuda_cnn_cublas.dll
cpp\Release\minimal_cuda_cnn_cublas.lib
cpp\Release\minimal_cuda_cnn_handmade.dll
cpp\Release\minimal_cuda_cnn_handmade.lib
```

cuBLAS variant 使用 `USE_CUBLAS=ON` 編譯，handmade variant 使用 `USE_CUBLAS=OFF`。

## 編譯單一 Variant

```powershell
.\scripts\build_windows_native.ps1 -Variant cublas
.\scripts\build_windows_native.ps1 -Variant handmade
.\scripts\build_windows_native.ps1 -Variant default
```

## GPU 架構

| GPU 系列 | `CMAKE_CUDA_ARCHITECTURES` |
|---|---:|
| RTX 20 系列 Turing | `75` |
| RTX 30 系列 Ampere | `86` |
| RTX 40 系列 Ada | `89` |
| RTX 50 系列 Blackwell | `120` |

## 手動 CMake 指令

```powershell
$CudaArch = "75"
cmake -S cpp -B cpp\build-windows-cublas -G "Visual Studio 16 2019" -A x64 `
  -DUSE_CUBLAS=ON `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_cublas `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-cublas --config Release --parallel
```

Handmade variant：

```powershell
$CudaArch = "75"
cmake -S cpp -B cpp\build-windows-handmade -G "Visual Studio 16 2019" -A x64 `
  -DUSE_CUBLAS=OFF `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_handmade `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-handmade --config Release --parallel
```

若切換 generator、platform、SDK 或 CUDA arch，先清掉舊 build 目錄。
`.\scripts\build_windows_native.ps1 -Clean` 會幫你處理。

## Python 載入說明

Python loader 在 Windows 上會把 native variant 對應到 `.dll` 檔名，並在呼叫 `ctypes.CDLL()` 前把 `cpp/` 與 `%CUDA_PATH%\bin` 加入 DLL 搜尋路徑。

C API 匯出介面保持一致，Windows loader 的對應關係為：

```text
runtime.cuda_variant=cublas   -> cpp\minimal_cuda_cnn_cublas.dll
runtime.cuda_variant=handmade -> cpp\minimal_cuda_cnn_handmade.dll
```

Python loader 在同一 process 切換 variant 時會重設 native handle cache，Windows smoke test 應依照 Linux 相同順序驗證兩個 DLL：

```powershell
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy runtime.cuda_variant=cublas train.epochs=1
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy runtime.cuda_variant=handmade train.epochs=1
```

和 Linux 一樣，建議先執行 `minicnn validate-dual-config` 與
`minicnn healthcheck`。它們現在會回傳 JSON-friendly 結果，或以簡短訊息失敗。

兩個 DLL variant 都應匯出 `maxpool_backward_nchw_status`，以及舊有的 void `maxpool_backward_nchw` symbol。

可把以下視為手動驗證完成的判定條件：

- `cpp\Release\` 下實際出現預期的 `.dll` 與 `.lib`
- 兩個 variant 都匯出 `maxpool_backward_nchw_status`
- `minicnn validate-dual-config` 與 `minicnn healthcheck` 通過
- `runtime.cuda_variant=cublas` 與 `runtime.cuda_variant=handmade` 都能各自完成 smoke run

## DLL 使用程式碼

repo 內現在也有可直接跑的 Python ctypes `.dll` smoke example：

```powershell
python -u examples\mnist_ctypes\check_native_library.py --variant handmade
python -u examples\mnist_ctypes\check_native_library.py --path cpp\minimal_cuda_cnn_cublas.dll
```

如果要看帶訓練流程的完整範例，使用：

```powershell
python -u examples\mnist_ctypes\train_mnist_so_full_cnn_frame.py --download
```

這個 smoke script 會走 MiniCNN 自己的 DLL resolver 與 binding layer，
先檢查必要 symbol，再做一次 GPU upload/download round-trip。
