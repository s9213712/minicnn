# Windows Native Build

> Status: unverified on a real Windows machine.
>
> This page is a build note, not a validated support claim. The repository
> includes the CMake settings and PowerShell helper needed to attempt Windows
> `.dll` builds, but this path is not covered by CI and has not been confirmed
> end to end in this environment.

This page documents the planned Windows build path for the native CUDA backend.

## Requirements

- Windows 10/11
- NVIDIA display driver with CUDA support
- Visual Studio 2022 with C++ workload
- CMake 3.20 or newer
- CUDA Toolkit installed on Windows
- PowerShell

## Build Both Native Variants

Run from the repository root in PowerShell:

```powershell
.\scripts\build_windows_native.ps1 -Variant both
```

For non-RTX-30 GPUs, pass a different CUDA architecture:

```powershell
.\scripts\build_windows_native.ps1 -Variant both -CudaArch 89
```

Expected outputs:

```text
cpp\minimal_cuda_cnn_cublas.dll
cpp\minimal_cuda_cnn_cublas.lib
cpp\minimal_cuda_cnn_handmade.dll
cpp\minimal_cuda_cnn_handmade.lib
```

The cuBLAS variant compiles with `USE_CUBLAS=ON`. The handmade variant compiles with `USE_CUBLAS=OFF`.

## Build One Variant

```powershell
.\scripts\build_windows_native.ps1 -Variant cublas
.\scripts\build_windows_native.ps1 -Variant handmade
.\scripts\build_windows_native.ps1 -Variant default
```

## Manual CMake Command

```powershell
$CudaArch = "86"
cmake -S cpp -B cpp\build-windows-cublas -G "Visual Studio 17 2022" -A x64 `
  -DUSE_CUBLAS=ON `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_cublas `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-cublas --config Release --parallel
```

For the handmade variant:

```powershell
$CudaArch = "86"
cmake -S cpp -B cpp\build-windows-handmade -G "Visual Studio 17 2022" -A x64 `
  -DUSE_CUBLAS=OFF `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_handmade `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-handmade --config Release --parallel
```

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

This part should be verified on a Windows machine after compiling the DLLs.

---

# Windows Native Build（中文）

> 狀態：尚未在實際 Windows 機器上驗證。
>
> 這份文件目前是建置備忘，不代表已正式支援。repo 內雖然已包含
> CMake 設定與 PowerShell helper，可嘗試產生 Windows `.dll`，
> 但這條路徑目前沒有 CI 驗證，也尚未在此環境完成端到端確認。

本文說明 Windows 平台的 native CUDA backend 建置流程。

## 需求

- Windows 10/11
- 支援 CUDA 的 NVIDIA 顯示驅動
- Visual Studio 2022（含 C++ 工作負載）
- CMake 3.20 以上
- Windows 端安裝的 CUDA Toolkit
- PowerShell

## 編譯兩種 Native Variant

在 PowerShell 專案根目錄執行：

```powershell
.\scripts\build_windows_native.ps1 -Variant both
```

若使用非 RTX 30 系列 GPU，傳入對應的 CUDA 架構：

```powershell
.\scripts\build_windows_native.ps1 -Variant both -CudaArch 89
```

預期輸出：

```text
cpp\minimal_cuda_cnn_cublas.dll
cpp\minimal_cuda_cnn_cublas.lib
cpp\minimal_cuda_cnn_handmade.dll
cpp\minimal_cuda_cnn_handmade.lib
```

cuBLAS variant 使用 `USE_CUBLAS=ON` 編譯，handmade variant 使用 `USE_CUBLAS=OFF`。

## 編譯單一 Variant

```powershell
.\scripts\build_windows_native.ps1 -Variant cublas
.\scripts\build_windows_native.ps1 -Variant handmade
.\scripts\build_windows_native.ps1 -Variant default
```

## 手動 CMake 指令

```powershell
$CudaArch = "86"
cmake -S cpp -B cpp\build-windows-cublas -G "Visual Studio 17 2022" -A x64 `
  -DUSE_CUBLAS=ON `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_cublas `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-cublas --config Release --parallel
```

Handmade variant：

```powershell
$CudaArch = "86"
cmake -S cpp -B cpp\build-windows-handmade -G "Visual Studio 17 2022" -A x64 `
  -DUSE_CUBLAS=OFF `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_handmade `
  -DCMAKE_CUDA_ARCHITECTURES=$CudaArch

cmake --build cpp\build-windows-handmade --config Release --parallel
```

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

此部分需在 Windows 機器上編譯 DLL 後實際驗證。
