# Windows native build

This page documents the planned Windows build path for the native CUDA backend. It is not validated in the current WSL environment, but the repository now includes the CMake settings and PowerShell helper needed to produce Windows `.dll` files.

## Requirements

- Windows 10/11
- NVIDIA display driver with CUDA support
- Visual Studio 2022 with C++ workload
- CMake 3.20 or newer
- CUDA Toolkit installed on Windows
- PowerShell

## Build both native variants

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

## Build one variant

```powershell
.\scripts\build_windows_native.ps1 -Variant cublas
.\scripts\build_windows_native.ps1 -Variant handmade
.\scripts\build_windows_native.ps1 -Variant default
```

## Manual CMake command

```powershell
cmake -S cpp -B cpp\build-windows-cublas -G "Visual Studio 17 2022" -A x64 `
  -DUSE_CUBLAS=ON `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_cublas `
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build cpp\build-windows-cublas --config Release --parallel
```

For the handmade variant:

```powershell
cmake -S cpp -B cpp\build-windows-handmade -G "Visual Studio 17 2022" -A x64 `
  -DUSE_CUBLAS=OFF `
  -DMINICNN_OUTPUT_NAME=minimal_cuda_cnn_handmade `
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build cpp\build-windows-handmade --config Release --parallel
```

## Python loading note

The Python loader maps native variants to `.dll` names on Windows and adds `cpp/` plus `%CUDA_PATH%\bin` to the DLL search path before `ctypes.CDLL()`.

The C API export surface is kept the same, so the intended Windows loader mapping is:

```text
runtime.cuda_variant=cublas   -> cpp\minimal_cuda_cnn_cublas.dll
runtime.cuda_variant=handmade -> cpp\minimal_cuda_cnn_handmade.dll
```

This part should be verified on a Windows machine after compiling the DLLs.
