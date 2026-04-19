from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from minicnn.paths import CPP_ROOT


REQUIRED_NATIVE_SYMBOLS = (
    'gpu_malloc',
    'gemm_forward',
    'conv_backward_precol',
    'softmax_xent_grad_loss_acc',
)


def build_native(use_cublas: bool = True, generator: str = "make", legacy_make: bool = False) -> None:
    if legacy_make:
        cmd = ['make', '-C', str(CPP_ROOT), f'USE_CUBLAS={1 if use_cublas else 0}']
        subprocess.run(cmd, check=True)
        return

    build_dir = CPP_ROOT / 'build'
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_generator = 'Unix Makefiles' if generator == 'make' else 'Ninja'
    configure_cmd = [
        'cmake', '-S', str(CPP_ROOT), '-B', str(build_dir),
        '-G', cmake_generator,
        f'-DUSE_CUBLAS={"ON" if use_cublas else "OFF"}',
    ]
    subprocess.run(configure_cmd, check=True)
    build_cmd = ['cmake', '--build', str(build_dir), '--parallel']
    subprocess.run(build_cmd, check=True)


def check_native() -> bool:
    so_path = CPP_ROOT / 'libminimal_cuda_cnn.so'
    cmake_so = CPP_ROOT / 'build' / 'libminimal_cuda_cnn.so'
    chosen = so_path if so_path.exists() else cmake_so
    if not chosen.exists():
        print(f"Native library not found. Build it with: minicnn build --check")
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
