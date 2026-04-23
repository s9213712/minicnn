param(
    [ValidateSet("default", "cublas", "handmade", "both")]
    [string]$Variant = "both",

    [string]$Generator = "Visual Studio 16 2019",
    [string]$Platform = "x64",
    [string]$Config = "Release",
    [string]$CudaArch = "75",

    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
$RepoRoot  = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir ".."))
$CppRoot   = [System.IO.Path]::Combine($RepoRoot, "cpp")

Write-Host "PSScriptRoot = [$PSScriptRoot]"
Write-Host "ScriptDir    = [$ScriptDir]"
Write-Host "RepoRoot     = [$RepoRoot]"
Write-Host "CppRoot      = [$CppRoot]"
Write-Host "Generator    = [$Generator]"
Write-Host "Platform     = [$Platform]"
Write-Host "Config       = [$Config]"
Write-Host "CudaArch     = [$CudaArch]"

function Get-RecommendedCudaArch {
    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        if (-not $gpuName) {
            return $null
        }
        $recommended = $null
        if ($gpuName -match 'RTX 20') {
            $recommended = '75'
        }
        elseif ($gpuName -match 'RTX 30') {
            $recommended = '86'
        }
        elseif ($gpuName -match 'RTX 40') {
            $recommended = '89'
        }
        elseif ($gpuName -match 'RTX 50') {
            $recommended = '120'
        }
        if ($recommended) {
            return @{
                Name = $gpuName
                CudaArch = $recommended
            }
        }
    }
    catch {
    }
    return $null
}

function Remove-BuildDir {
    param([string]$Dir)

    if (Test-Path $Dir) {
        Write-Host "Removing build dir: [$Dir]"
        Remove-Item -Recurse -Force $Dir
    }
}

function Invoke-CMake {
    param(
        [string[]]$ArgsList,
        [string]$FailMessage,
        [string]$BuildDir,
        [string]$VariantName
    )

    Write-Host "cmake args:"
    $ArgsList | ForEach-Object { Write-Host "  $_" }

    try {
        & cmake @ArgsList
        if ($LASTEXITCODE -ne 0) {
            throw "$FailMessage (exit code $LASTEXITCODE)"
        }
    }
    catch {
        Write-Host ""
        Write-Host "[FAILED] $VariantName" -ForegroundColor Red
        Write-Host "Reason: $($_.Exception.Message)" -ForegroundColor Yellow
        if ($BuildDir) {
            Write-Host "BuildDir: $BuildDir" -ForegroundColor DarkYellow
        }
        exit 1
    }
}

function Get-DllLocations {
    param(
        [string]$BuildDir,
        [string]$OutputName
    )

    $results = @()
    foreach ($root in @($BuildDir, $CppRoot, (Join-Path $CppRoot "Release"))) {
        if (Test-Path $root) {
            $results += Get-ChildItem -Path $root -Recurse -Filter "$OutputName.dll" -ErrorAction SilentlyContinue
            $results += Get-ChildItem -Path $root -Recurse -Filter "*.dll" -ErrorAction SilentlyContinue |
                Where-Object { $_.BaseName -eq $OutputName }
        }
    }

    $results |
        Sort-Object FullName -Unique |
        ForEach-Object { $_.FullName }
}

function Get-LibLocations {
    param(
        [string]$BuildDir,
        [string]$OutputName
    )

    $results = @()
    foreach ($root in @($BuildDir, $CppRoot, (Join-Path $CppRoot "Release"))) {
        if (Test-Path $root) {
            $results += Get-ChildItem -Path $root -Recurse -Filter "$OutputName.lib" -ErrorAction SilentlyContinue
            $results += Get-ChildItem -Path $root -Recurse -Filter "*.lib" -ErrorAction SilentlyContinue |
                Where-Object { $_.BaseName -eq $OutputName }
        }
    }

    $results |
        Sort-Object FullName -Unique |
        ForEach-Object { $_.FullName }
}

function Stage-LoaderArtifacts {
    param(
        [string[]]$Dlls,
        [string[]]$Libs
    )

    $staged = @()
    foreach ($path in @($Dlls + $Libs)) {
        if (-not $path) {
            continue
        }
        $sourcePath = [System.IO.Path]::GetFullPath($path)
        $destPath = [System.IO.Path]::Combine($CppRoot, [System.IO.Path]::GetFileName($sourcePath))
        if ($sourcePath -ne $destPath) {
            Copy-Item -Path $sourcePath -Destination $destPath -Force
        }
        $staged += $destPath
    }
    $staged | Sort-Object -Unique
}

function Report-Success {
    param(
        [string]$VariantName,
        [string]$BuildDir,
        [string]$OutputName
    )

    $dlls = @(Get-DllLocations -BuildDir $BuildDir -OutputName $OutputName)
    $libs = @(Get-LibLocations -BuildDir $BuildDir -OutputName $OutputName)
    $staged = @(Stage-LoaderArtifacts -Dlls $dlls -Libs $libs)

    Write-Host ""
    if ($dlls.Count -gt 0) {
        Write-Host "Build SUCCESS ($VariantName)" -ForegroundColor Green
        Write-Host ""
        Write-Host "DLL location:" -ForegroundColor Green
        $dlls | ForEach-Object { Write-Host "  $_" }
        if ($libs.Count -gt 0) {
            Write-Host ""
            Write-Host "LIB location:" -ForegroundColor Green
            $libs | ForEach-Object { Write-Host "  $_" }
        }
        if ($staged.Count -gt 0) {
            Write-Host ""
            Write-Host "Loader-ready copies:" -ForegroundColor Green
            $staged | ForEach-Object { Write-Host "  $_" }
        }
        Write-Host ""
        Write-Host "Next step:" -ForegroundColor Cyan
        Write-Host "  minicnn validate-dual-config --config configs/dual_backend_cnn.yaml"
        Write-Host "  minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy runtime.cuda_variant=$VariantName"
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "  - If DLL not found, check the printed build directory and Visual Studio config."
        Write-Host "  - If the DLL cannot load, check CUDA_PATH and the NVIDIA driver/runtime."
    }
    else {
        Write-Host "[SUCCESS] $VariantName build completed, but no DLL named [$OutputName.dll] was found." -ForegroundColor Yellow
        Write-Host "Checked build dir: $BuildDir" -ForegroundColor DarkYellow
    }
}

function Build-One {
    param(
        [string]$Name,
        [bool]$UseCublas
    )

    $BuildDir = [System.IO.Path]::Combine($CppRoot, "build-windows-$Name")
    $OutputName = if ($Name -eq "default") { "minimal_cuda_cnn" } else { "minimal_cuda_cnn_$Name" }
    $UseCublasValue = if ($UseCublas) { "ON" } else { "OFF" }

    Write-Host "BuildDir     = [$BuildDir]"
    Write-Host "OutputName   = [$OutputName]"
    Write-Host "UseCublas    = [$UseCublasValue]"

    if ($Clean) {
        Remove-BuildDir -Dir $BuildDir
    }

    $ConfigureArgs = @(
        "-S", $CppRoot,
        "-B", $BuildDir,
        "-G", $Generator,
        "-A", $Platform,
        "-DUSE_CUBLAS=$UseCublasValue",
        "-DMINICNN_OUTPUT_NAME=$OutputName",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    )

    Invoke-CMake -ArgsList $ConfigureArgs -FailMessage "CMake configure failed for variant: $Name" -BuildDir $BuildDir -VariantName $Name

    $BuildArgs = @(
        "--build", $BuildDir,
        "--config", $Config,
        "--parallel"
    )

    Invoke-CMake -ArgsList $BuildArgs -FailMessage "CMake build failed for variant: $Name" -BuildDir $BuildDir -VariantName $Name

    Report-Success -VariantName $Name -BuildDir $BuildDir -OutputName $OutputName
}

$gpuRecommendation = Get-RecommendedCudaArch
if ($gpuRecommendation) {
    Write-Host "Detected GPU: [$($gpuRecommendation.Name)]"
    Write-Host "Recommended CudaArch: [$($gpuRecommendation.CudaArch)]"
}

switch ($Variant) {
    "both" {
        Build-One -Name "cublas" -UseCublas $true
        Build-One -Name "handmade" -UseCublas $false
    }
    "cublas" {
        Build-One -Name "cublas" -UseCublas $true
    }
    "handmade" {
        Build-One -Name "handmade" -UseCublas $false
    }
    "default" {
        Build-One -Name "default" -UseCublas $true
    }
    default {
        throw "Unknown variant: $Variant"
    }
}
