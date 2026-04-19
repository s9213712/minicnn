param(
    [ValidateSet("default", "cublas", "handmade", "both")]
    [string]$Variant = "both",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Platform = "x64",
    [string]$Config = "Release",
    [string]$CudaArch = "86"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$CppRoot = Join-Path $RepoRoot "cpp"

function Build-One {
    param(
        [string]$Name,
        [bool]$UseCublas
    )

    $BuildDir = Join-Path $CppRoot "build-windows-$Name"
    $OutputName = if ($Name -eq "default") { "minimal_cuda_cnn" } else { "minimal_cuda_cnn_$Name" }
    $UseCublasValue = if ($UseCublas) { "ON" } else { "OFF" }

    cmake -S $CppRoot -B $BuildDir -G $Generator -A $Platform `
        "-DUSE_CUBLAS=$UseCublasValue" `
        "-DMINICNN_OUTPUT_NAME=$OutputName" `
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    cmake --build $BuildDir --config $Config --parallel
}

switch ($Variant) {
    "both" {
        Build-One -Name "cublas" -UseCublas $true
        Build-One -Name "handmade" -UseCublas $false
    }
    "cublas" { Build-One -Name "cublas" -UseCublas $true }
    "handmade" { Build-One -Name "handmade" -UseCublas $false }
    "default" { Build-One -Name "default" -UseCublas $true }
}

Write-Host "Windows native build finished. Check cpp/*.dll and cpp/*.lib."
