from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TEMPLATE = "templates/cifar10/convnext_explicit_cuda_native_smoke.yaml"


def _run_native_variant(tmp_path: Path, variant: str, *overrides: str) -> tuple[dict, dict]:
    artifacts_root = tmp_path / variant
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"src:{existing_pythonpath}" if existing_pythonpath else "src"
    )

    cmd = [
        "python3",
        "-m",
        "minicnn.cli",
        "train-native",
        "--config",
        SMOKE_TEMPLATE,
        "project.name=tolerance-matrix",
        f"project.artifacts_root={artifacts_root}",
        "train.seed=123",
        "dataset.num_samples=8",
        "dataset.val_samples=4",
        "train.epochs=1",
        *overrides,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    summary_paths = list(artifacts_root.rglob("summary.json"))
    assert len(summary_paths) == 1
    summary_path = summary_paths[0]
    metrics_path = summary_path.with_name("metrics.jsonl")
    assert metrics_path.exists()

    summary = json.loads(summary_path.read_text())
    metrics_rows = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    assert metrics_rows
    return summary, metrics_rows[-1]


def test_cuda_native_tolerance_matrix_stays_within_bounds(tmp_path: Path) -> None:
    fp32_summary, fp32_last = _run_native_variant(
        tmp_path,
        "fp32",
        "train.batch_size=4",
        "train.grad_accum_steps=1",
        "train.amp=false",
    )
    amp_summary, amp_last = _run_native_variant(
        tmp_path,
        "amp",
        "train.batch_size=4",
        "train.grad_accum_steps=1",
        "train.amp=true",
        "train.amp_loss_scale=128",
    )
    accum_summary, accum_last = _run_native_variant(
        tmp_path,
        "grad-accum",
        "train.batch_size=2",
        "train.grad_accum_steps=2",
        "train.amp=false",
    )

    for summary, last_row in (
        (fp32_summary, fp32_last),
        (amp_summary, amp_last),
        (accum_summary, accum_last),
    ):
        assert summary["schema_name"] == "minicnn.cuda_native.training.summary"
        assert summary["schema_version"] == 1
        assert summary["artifact_kind"] == "training_run_summary"
        assert isinstance(last_row["train_loss"], (int, float))
        assert isinstance(last_row["val_loss"], (int, float))
        assert isinstance(last_row["val_acc"], (int, float))

    assert amp_summary["amp_runtime"]["loss_scale"] >= 1.0
    assert amp_last["amp"]["loss_scale"] >= 1.0
    assert accum_summary["optimizer_runtime"]["grad_buffer_reuses"] >= 1
    assert accum_last["optimizer_runtime"]["grad_buffer_reuses_epoch"] >= 1

    assert abs(fp32_last["train_loss"] - amp_last["train_loss"]) <= 0.75
    assert abs(fp32_last["val_loss"] - amp_last["val_loss"]) <= 0.75
    assert abs(fp32_last["val_acc"] - amp_last["val_acc"]) <= 25.0

    assert abs(fp32_last["train_loss"] - accum_last["train_loss"]) <= 0.5
    assert abs(fp32_last["val_loss"] - accum_last["val_loss"]) <= 0.5
    assert abs(fp32_last["val_acc"] - accum_last["val_acc"]) <= 25.0
