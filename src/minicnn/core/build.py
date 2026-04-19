from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path

from minicnn.paths import CPP_ROOT


if os.name == 'nt':
    NATIVE_LIBRARY_NAMES = {
        'default': 'minimal_cuda_cnn.dll',
        'cublas': 'minimal_cuda_cnn_cublas.dll',
        'handmade': 'minimal_cuda_cnn_handmade.dll',
    }
else:
    NATIVE_LIBRARY_NAMES = {
        'default': 'libminimal_cuda_cnn.so',
        'cublas': 'libminimal_cuda_cnn_cublas.so',
        'handmade': 'libminimal_cuda_cnn_handmade.so',
    }

REQUIRED_NATIVE_SYMBOLS = (
    'gpu_malloc',
    'gemm_forward',
    'conv_backward_precol',
    'softmax_xent_grad_loss_acc',
)


def native_library_path(variant: str = 'default') -> Path:
    try:
        return CPP_ROOT / NATIVE_LIBRARY_NAMES[variant]
    except KeyError as exc:
        choices = ', '.join(sorted(NATIVE_LIBRARY_NAMES))
        raise ValueError(f'Unknown native variant {variant!r}; expected one of: {choices}, both') from exc


def _build_one_cmake(use_cublas: bool, generator: str, variant: str) -> None:
    build_dir = CPP_ROOT / ('build-cmake' if variant == 'default' else f'build-cmake-{variant}')
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_generator = 'Unix Makefiles' if generator == 'make' else 'Ninja'
    output_name = native_library_path(variant).name
    if output_name.startswith('lib') and output_name.endswith('.so'):
        output_name = output_name[3:-3]
    elif output_name.endswith('.dll'):
        output_name = output_name[:-4]
    cuda_home = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    configure_cmd = [
        'cmake', '-S', str(CPP_ROOT), '-B', str(build_dir),
        '-G', cmake_generator,
        f'-DUSE_CUBLAS={"ON" if use_cublas else "OFF"}',
        f'-DMINICNN_OUTPUT_NAME={output_name}',
        '-DCMAKE_CUDA_ARCHITECTURES=86',
    ]
    nvcc = cuda_home / 'bin' / ('nvcc.exe' if os.name == 'nt' else 'nvcc')
    if nvcc.exists():
        configure_cmd.extend([
            f'-DCMAKE_CUDA_COMPILER={nvcc}',
            f'-DCUDAToolkit_ROOT={cuda_home}',
        ])
    subprocess.run(configure_cmd, check=True)
    build_cmd = ['cmake', '--build', str(build_dir), '--parallel']
    subprocess.run(build_cmd, check=True)


def build_native(
    use_cublas: bool = True,
    generator: str = "make",
    legacy_make: bool = False,
    variant: str = 'default',
) -> None:
    if legacy_make:
        if variant == 'both':
            cmd = ['make', '-C', str(CPP_ROOT), 'variants']
        elif variant in {'cublas', 'handmade'}:
            cmd = ['make', '-C', str(CPP_ROOT), variant]
        else:
            cmd = ['make', '-C', str(CPP_ROOT), f'USE_CUBLAS={1 if use_cublas else 0}']
        subprocess.run(cmd, check=True)
        return

    if variant == 'both':
        _build_one_cmake(True, generator, 'cublas')
        _build_one_cmake(False, generator, 'handmade')
    elif variant == 'cublas':
        _build_one_cmake(True, generator, 'cublas')
    elif variant == 'handmade':
        _build_one_cmake(False, generator, 'handmade')
    else:
        _build_one_cmake(use_cublas, generator, 'default')


def check_native(variant: str = 'default') -> bool:
    if variant == 'both':
        return check_native('cublas') and check_native('handmade')
    so_path = native_library_path(variant)
    cmake_so = CPP_ROOT / 'build' / so_path.name
    chosen = so_path if so_path.exists() else cmake_so
    if not chosen.exists():
        print(f"Native library not found: {chosen}. Build it with: minicnn build --check")
        return False
    if not shutil.which('nm'):
        print(f"Found native library: {chosen}")
        return True

    result = subprocess.run(['nm', '-D', str(chosen)], check=True, text=True, capture_output=True)
    missing = [symbol for symbol in REQUIRED_NATIVE_SYMBOLS if symbol not in result.stdout]
    if missing:
        raise RuntimeError(f"Native library is missing required symbols: {missing}")
    print(f"Found native library: {chosen}; required symbols OK")
    return True
