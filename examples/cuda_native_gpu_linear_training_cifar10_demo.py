"""Run one native GPU Linear + SoftmaxCE + SGD step on a CIFAR-10 batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minicnn.cuda_native import native_gpu_linear_training_step
from minicnn.data.cifar10 import cifar10_ready, load_cifar10_test, normalize_cifar


def _load_batch(
    data_root: str,
    batch_size: int,
    *,
    allow_random_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    root = Path(data_root)
    if cifar10_ready(root):
        x_test, y_test = load_cifar10_test(root, download=False)
        x = normalize_cifar(x_test[:batch_size]).reshape(batch_size, -1).astype(np.float32)
        y = np.asarray(y_test[:batch_size], dtype=np.int32)
        return x, y, "official:cifar10:test_batch"
    if not allow_random_fallback:
        raise FileNotFoundError(
            f"CIFAR-10 data is missing under {root}. Run `minicnn prepare-data` or pass "
            "`--allow-random-fallback` for a deterministic non-dataset smoke."
        )
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(batch_size, 3 * 32 * 32)).astype(np.float32)
    y = rng.integers(0, 10, size=(batch_size,), dtype=np.int32)
    return x, y, "random:fallback"


def _reference_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)
    loss_sum = float(-np.log(probs[np.arange(labels.shape[0]), labels] + 1e-10).sum())
    return weight - lr * grad_weight, bias - lr * grad_bias, loss_sum / float(labels.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cuda_native native GPU linear training step on a CIFAR-10 batch."
    )
    parser.add_argument("--data-root", default="data/cifar-10-batches-py")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--allow-random-fallback", action="store_true")
    args = parser.parse_args()

    batch_size = max(1, int(args.batch_size))
    x, labels, dataset_source = _load_batch(
        args.data_root,
        batch_size,
        allow_random_fallback=bool(args.allow_random_fallback),
    )
    rng = np.random.default_rng(int(args.seed))
    weight = rng.normal(0.0, 0.01, size=(10, x.shape[1])).astype(np.float32)
    bias = np.zeros((10,), dtype=np.float32)
    result = native_gpu_linear_training_step(x, labels, weight, bias, lr=float(args.lr))
    ref_weight, ref_bias, ref_loss = _reference_step(x, labels, weight, bias, float(args.lr))
    payload = {
        "demo": "cuda_native_gpu_linear_training_cifar10",
        "dataset_source": dataset_source,
        "batch_shape": list(x.shape),
        "loss_mean": result.loss_mean,
        "reference_loss_mean": ref_loss,
        "max_abs_updated_weight_diff_vs_reference": float(np.max(np.abs(result.updated_weight - ref_weight))),
        "max_abs_updated_bias_diff_vs_reference": float(np.max(np.abs(result.updated_bias - ref_bias))),
        "correct_count": result.correct_count,
        "native_execution_kinds": {
            key: value
            for key, value in result.runtime_summary["execution_kinds"].items()
            if key.startswith("gpu_native_train:")
        },
        "device_pointer_allocation_events": result.runtime_summary["device_pointer_allocation_events"],
        "device_sync_to_device_events": result.runtime_summary["device_sync_to_device_events"],
        "device_sync_to_host_events": result.runtime_summary["device_sync_to_host_events"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
