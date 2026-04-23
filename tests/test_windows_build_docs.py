from __future__ import annotations

from pathlib import Path


def test_windows_guide_uses_build_windows_output_paths():
    guide = (Path(__file__).resolve().parents[1] / 'docs' / 'guide_windows_build.md').read_text(encoding='utf-8')

    assert 'build-windows-cublas\\Release\\minimal_cuda_cnn_cublas.dll' in guide
    assert 'build-windows-handmade\\Release\\minimal_cuda_cnn_handmade.dll' in guide
    assert 'cpp\\Release\\minimal_cuda_cnn_cublas.dll' not in guide
    assert 'cpp\\Release\\minimal_cuda_cnn_handmade.dll' not in guide


def test_windows_build_script_reports_success_locations_and_next_steps():
    script = (Path(__file__).resolve().parents[1] / 'scripts' / 'build_windows_native.ps1').read_text(encoding='utf-8')

    assert 'Build SUCCESS' in script
    assert 'DLL location:' in script
    assert 'Next step:' in script
    assert 'Troubleshooting:' in script
    assert 'Detected GPU:' in script
    assert 'Recommended CudaArch:' in script
