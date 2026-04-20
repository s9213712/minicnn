import importlib
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {**os.environ, 'PYTHONPATH': str(REPO_ROOT / 'src')}


def test_cuda_backend_import_is_lazy_without_library():
    code = (
        "import os\n"
        "os.environ['MINICNN_CUDA_SO'] = '/tmp/minicnn_missing_library.so'\n"
        "import minicnn.core.cuda_backend as cb\n"
        "print(cb.resolve_library_path())\n"
        "try:\n"
        "    cb.get_lib()\n"
        "except RuntimeError as exc:\n"
        "    print('lazy-error-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        text=True,
        capture_output=True,
        check=True,
        env=SUBPROCESS_ENV,
    )
    assert '/tmp/minicnn_missing_library.so' in result.stdout
    assert 'lazy-error-ok' in result.stdout


def test_cuda_backend_reset_library_cache():
    import minicnn.core.cuda_backend as cuda_backend

    marker = object()
    cuda_backend._lib = marker

    cuda_backend.reset_library_cache()

    assert cuda_backend._lib is None


def test_train_cuda_import_does_not_load_missing_library():
    code = (
        "import os\n"
        "os.environ['MINICNN_CUDA_SO'] = '/tmp/minicnn_missing_library.so'\n"
        "import minicnn.training.train_cuda\n"
        "print('import-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        text=True,
        capture_output=True,
        check=True,
        env=SUBPROCESS_ENV,
    )
    assert 'import-ok' in result.stdout


def test_settings_shape_fields_match_model_geometry():
    from minicnn.config import settings

    assert settings.FC_IN == settings.C4_OUT * settings.P2H * settings.P2W
    assert settings.C2_IN == settings.C1_OUT
    assert settings.C3_IN == settings.C2_OUT
    assert settings.C4_IN == settings.C3_OUT


def test_settings_env_overrides(monkeypatch):
    monkeypatch.setenv('MINICNN_BATCH', '7')
    monkeypatch.setenv('MINICNN_EPOCHS', '3')
    monkeypatch.setenv('MINICNN_RANDOM_CROP_PADDING', '4')
    monkeypatch.setenv('MINICNN_HORIZONTAL_FLIP', 'false')
    from minicnn.config import settings

    importlib.reload(settings)

    assert settings.BATCH == 7
    assert settings.EPOCHS == 3
    assert settings.RANDOM_CROP_PADDING == 4
    assert settings.HORIZONTAL_FLIP is False

    monkeypatch.delenv('MINICNN_BATCH')
    monkeypatch.delenv('MINICNN_EPOCHS')
    monkeypatch.delenv('MINICNN_RANDOM_CROP_PADDING')
    monkeypatch.delenv('MINICNN_HORIZONTAL_FLIP')
    importlib.reload(settings)
