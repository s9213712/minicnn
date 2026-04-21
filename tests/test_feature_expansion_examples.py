"""Integration tests that run each feature_expansion example script."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "feature_expansion"


def _run(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def test_01_activations():
    r = _run("01_activations.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "All activations ran successfully" in r.stdout


def test_02_optimizers():
    r = _run("02_optimizers.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "All optimizers ran successfully" in r.stdout


def test_03_schedulers():
    r = _run("03_schedulers.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "All schedulers ran successfully" in r.stdout


def test_04_initialization():
    r = _run("04_initialization.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "All initializers ran successfully" in r.stdout


def test_05_label_smoothing():
    r = _run("05_label_smoothing.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "Label smoothing demo ran successfully" in r.stdout


def test_06_train_autograd_enhanced():
    r = _run("06_train_autograd_enhanced.py", timeout=120)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "Training completed successfully" in r.stdout


def test_07_flex_presets():
    torch = pytest.importorskip("torch")
    r = _run("07_flex_presets.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "Flex presets demo ran successfully" in r.stdout


def test_08_augmentation():
    torch = pytest.importorskip("torch")
    r = _run("08_augmentation.py")
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "Augmentation demo ran successfully" in r.stdout
