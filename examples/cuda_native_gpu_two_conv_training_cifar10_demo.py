"""Run one native GPU two-Conv/ReLU/MaxPool/Linear training step on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minicnn.cuda_native import native_gpu_two_conv_relu_pool_linear_training_step
from minicnn.data.cifar10 import cifar10_ready, load_cifar10_test, normalize_cifar


def _load_batch(data_root: str, batch_size: int, *, allow_random_fallback: bool) -> tuple[np.ndarray, np.ndarray, str]:
    root = Path(data_root)
    if cifar10_ready(root):
        x_test, y_test = load_cifar10_test(root, download=False)
        x = normalize_cifar(x_test[:batch_size]).astype(np.float32)
        y = np.asarray(y_test[:batch_size], dtype=np.int32)
        return x, y, "official:cifar10:test_batch"
    if not allow_random_fallback:
        raise FileNotFoundError(
            f"CIFAR-10 data is missing under {root}. Run `minicnn prepare-data` or pass "
            "`--allow-random-fallback` for a deterministic non-dataset smoke."
        )
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(batch_size, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, size=(batch_size,), dtype=np.int32)
    return x, y, "random:fallback"


def _conv_valid(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, in_c, height, width = x.shape
    out_c, _, kh, kw = weight.shape
    out_h = height - kh + 1
    out_w = width - kw + 1
    out = np.zeros((n, out_c, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    out[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + kh, ow:ow + kw] * weight[oc])
    return out


def _maxpool2x2(x: np.ndarray) -> np.ndarray:
    n, channels, height, width = x.shape
    out = np.zeros((n, channels, height // 2, width // 2), dtype=np.float32)
    for ni in range(n):
        for ci in range(channels):
            for oh in range(height // 2):
                for ow in range(width // 2):
                    out[ni, ci, oh, ow] = np.max(x[ni, ci, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2])
    return out


def _maxpool2x2_backward(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for ci in range(x.shape[1]):
            for oh in range(grad_out.shape[2]):
                for ow in range(grad_out.shape[3]):
                    window = x[ni, ci, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                    flat_idx = int(np.argmax(window))
                    ih = oh * 2 + flat_idx // 2
                    iw = ow * 2 + flat_idx % 2
                    grad[ni, ci, ih, iw] += grad_out[ni, ci, oh, ow]
    return grad


def _conv_valid_backward(grad_out: np.ndarray, x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_weight = np.zeros_like(weight)
    grad_input = np.zeros_like(x)
    _, _, kh, kw = weight.shape
    for ni in range(x.shape[0]):
        for oc in range(weight.shape[0]):
            for oh in range(grad_out.shape[2]):
                for ow in range(grad_out.shape[3]):
                    grad_val = grad_out[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(kh):
                            for s in range(kw):
                                grad_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += weight[oc, ci, r, s] * grad_val
    return grad_weight, grad_input


def _reference_step(
    x: np.ndarray,
    labels: np.ndarray,
    conv1_weight: np.ndarray,
    conv2_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    conv1_pre = _conv_valid(x, conv1_weight)
    conv1 = np.maximum(conv1_pre, 0.0)
    conv2_pre = _conv_valid(conv1, conv2_weight)
    conv2 = np.maximum(conv2_pre, 0.0)
    pooled = _maxpool2x2(conv2)
    flat = pooled.reshape(x.shape[0], -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ linear_weight).reshape(pooled.shape)
    grad_conv2 = _maxpool2x2_backward(grad_pooled, conv2)
    grad_conv2 = np.where(conv2 > 0.0, grad_conv2, 0.0)
    grad_conv2_weight, grad_conv1 = _conv_valid_backward(grad_conv2, conv1, conv2_weight)
    grad_conv1 = np.where(conv1 > 0.0, grad_conv1, 0.0)
    grad_conv1_weight, _grad_input = _conv_valid_backward(grad_conv1, x, conv1_weight)
    loss_sum = float(-np.log(probs[np.arange(labels.shape[0]), labels] + 1e-10).sum())
    return (
        conv1_weight - lr * grad_conv1_weight,
        conv2_weight - lr * grad_conv2_weight,
        linear_weight - lr * grad_linear_weight,
        linear_bias - lr * grad_linear_bias,
        loss_sum / float(labels.shape[0]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cuda_native native GPU two-conv training step on a CIFAR-10 batch."
    )
    parser.add_argument("--data-root", default="data/cifar-10-batches-py")
    parser.add_argument("--batch-size", type=int, default=2)
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
    conv1_weight = rng.normal(0.0, 0.03, size=(4, 3, 3, 3)).astype(np.float32)
    conv2_weight = rng.normal(0.0, 0.03, size=(6, 4, 3, 3)).astype(np.float32)
    linear_weight = rng.normal(0.0, 0.01, size=(10, 6 * 14 * 14)).astype(np.float32)
    linear_bias = np.zeros((10,), dtype=np.float32)

    result = native_gpu_two_conv_relu_pool_linear_training_step(
        x,
        labels,
        conv1_weight,
        conv2_weight,
        linear_weight,
        linear_bias,
        lr=float(args.lr),
    )
    ref_conv1, ref_conv2, ref_linear_weight, ref_linear_bias, ref_loss = _reference_step(
        x,
        labels,
        conv1_weight,
        conv2_weight,
        linear_weight,
        linear_bias,
        float(args.lr),
    )
    payload = {
        "demo": "cuda_native_gpu_two_conv_training_cifar10",
        "dataset_source": dataset_source,
        "batch_shape": list(x.shape),
        "loss_mean": result.loss_mean,
        "reference_loss_mean": ref_loss,
        "max_abs_updated_conv1_weight_diff_vs_reference": float(np.max(np.abs(result.updated_conv1_weight - ref_conv1))),
        "max_abs_updated_conv2_weight_diff_vs_reference": float(np.max(np.abs(result.updated_conv2_weight - ref_conv2))),
        "max_abs_updated_linear_weight_diff_vs_reference": float(np.max(np.abs(result.updated_linear_weight - ref_linear_weight))),
        "max_abs_updated_linear_bias_diff_vs_reference": float(np.max(np.abs(result.updated_linear_bias - ref_linear_bias))),
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
