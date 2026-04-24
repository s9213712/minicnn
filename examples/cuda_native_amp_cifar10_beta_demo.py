from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.data.cifar10 import load_cifar10_test, normalize_cifar
from minicnn.unified._cuda_native_support import evaluate_native_graph, resolve_loss_type
from minicnn.unified.trainer import train_unified_from_config


def build_demo_config(*, data_root: str, artifacts_root: str, train_samples: int, val_samples: int) -> dict:
    return {
        "project": {
            "name": "minicnn",
            "run_name": "cuda-native-amp-cifar10-beta-demo",
            "artifacts_root": artifacts_root,
        },
        "engine": {
            "backend": "cuda_native",
            "strict_backend_validation": True,
            "planner_strategy": "reuse",
        },
        "dataset": {
            "type": "cifar10",
            "data_root": data_root,
            "download": False,
            "num_samples": train_samples,
            "val_samples": val_samples,
            "num_classes": 10,
            "input_shape": [3, 32, 32],
            "seed": 42,
        },
        "model": {
            "layers": [
                {"type": "Flatten"},
                {"type": "Linear", "out_features": 10},
            ],
        },
        "train": {
            "epochs": 1,
            "batch_size": 128,
            "device": "cpu",
            "amp": True,
            "amp_loss_scale": 128.0,
            "grad_accum_steps": 2,
            "num_workers": 0,
            "log_every": 10,
            "init_seed": 42,
        },
        "loss": {
            "type": "CrossEntropyLoss",
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 0.005,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate a beta-grade cuda_native AMP demo on real CIFAR-10.")
    parser.add_argument("--data-root", default="data/cifar-10-batches-py")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    args = parser.parse_args()

    cfg = build_demo_config(
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )
    run_dir = train_unified_from_config(cfg)
    summary_path = Path(run_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    best_model_path = Path(summary["best_model_path"])
    with np.load(best_model_path, allow_pickle=False) as checkpoint:
        params = {
            key: checkpoint[key]
            for key in checkpoint.files
        }

    input_shape = tuple(int(dim) for dim in cfg["dataset"]["input_shape"])
    graph = build_cuda_native_graph(cfg["model"], (args.eval_batch_size, *input_shape))
    x_test, y_test = load_cifar10_test(args.data_root, download=False)
    test_metrics = evaluate_native_graph(
        graph,
        normalize_cifar(x_test),
        y_test.astype(np.int64),
        params,
        args.eval_batch_size,
        resolve_loss_type(cfg["loss"]),
        amp_enabled=bool(cfg["train"]["amp"]),
    )

    payload = {
        "status": "ok",
        "backend": "cuda_native",
        "summary_status": summary["capabilities"]["summary_status"],
        "support_tier_assessment": summary["support_tier_assessment"],
        "run_dir": str(run_dir),
        "best_model_path": str(best_model_path),
        "best_val_acc": float(summary["best_val_acc"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "train_samples": int(args.train_samples),
        "val_samples": int(args.val_samples),
        "test_samples": int(x_test.shape[0]),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
