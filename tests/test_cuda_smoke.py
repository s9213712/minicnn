import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


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


def test_train_cuda_import_does_not_create_run_dir(tmp_path):
    run_dir = tmp_path / 'legacy-run-dir'
    code = (
        "import os\n"
        f"os.environ['MINICNN_ARTIFACT_RUN_DIR'] = {str(run_dir)!r}\n"
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
    assert not run_dir.exists()


def test_train_torch_baseline_import_does_not_create_run_dir(tmp_path):
    run_dir = tmp_path / 'torch-baseline-run-dir'
    code = (
        "import os\n"
        f"os.environ['MINICNN_ARTIFACT_RUN_DIR'] = {str(run_dir)!r}\n"
        "import minicnn.training.train_torch_baseline\n"
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
    assert not run_dir.exists()


def test_native_library_smoke_example_help():
    script = REPO_ROOT / 'examples' / 'mnist_ctypes' / 'check_native_library.py'
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        text=True,
        capture_output=True,
        check=True,
        env=SUBPROCESS_ENV,
    )
    assert '--variant' in result.stdout
    assert '--path' in result.stdout
    assert 'Smoke-test MiniCNN native CUDA library loading' in result.stdout


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


def test_settings_repeated_apply_updates_legacy_snapshot():
    from minicnn.config import settings
    from minicnn.config.schema import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.train.batch_size = 16
    settings.apply_experiment_config(cfg)
    first_snapshot = dict(settings.legacy_values())

    cfg2 = ExperimentConfig()
    cfg2.train.batch_size = 32
    settings.apply_experiment_config(cfg2)
    second_snapshot = dict(settings.legacy_values())

    assert first_snapshot['BATCH'] == 16
    assert second_snapshot['BATCH'] == 32
    assert settings.BATCH == 32


def test_settings_summarize_includes_override_provenance(monkeypatch):
    monkeypatch.setenv('MINICNN_BATCH', '11')
    from minicnn.config import settings

    importlib.reload(settings)

    assert settings.override_provenance()['BATCH'] == 'MINICNN_BATCH'
    assert settings.summarize()['override_provenance']['BATCH'] == 'MINICNN_BATCH'

    monkeypatch.delenv('MINICNN_BATCH')
    importlib.reload(settings)


def test_settings_get_arch_requires_apply(monkeypatch):
    from minicnn.config import settings

    monkeypatch.setattr(settings, '_arch', None)

    with pytest.raises(RuntimeError, match='apply_experiment_config'):
        settings.get_arch()
