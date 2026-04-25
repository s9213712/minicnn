"""Run one native GPU ConvNeXt-style bridge training step on a CIFAR-10 batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minicnn.cuda_native import native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step
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


def _gelu(values: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
    return (0.5 * values * (1.0 + np.tanh(inner))).astype(np.float32)


def _gelu_grad(values: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
    tanh_inner = np.tanh(inner)
    left = 0.5 * (1.0 + tanh_inner)
    right = 0.5 * values * (1.0 - tanh_inner ** 2) * np.sqrt(2.0 / np.pi) * (
        1.0 + 3.0 * 0.044715 * values ** 2
    )
    return (left + right).astype(np.float32)


def _depthwise_valid(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, in_c, height, width = x.shape
    out_c, _, kh, kw = weight.shape
    out_h = height - kh + 1
    out_w = width - kw + 1
    multiplier = out_c // in_c
    out = np.zeros((n, out_c, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for oc in range(out_c):
            ic = oc // multiplier
            for oh in range(out_h):
                for ow in range(out_w):
                    out[ni, oc, oh, ow] = np.sum(x[ni, ic, oh:oh + kh, ow:ow + kw] * weight[oc, 0])
    return out


def _depthwise_backward(grad_out: np.ndarray, x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_weight = np.zeros_like(weight)
    grad_input = np.zeros_like(x)
    out_c, _, kh, kw = weight.shape
    multiplier = out_c // x.shape[1]
    for ni in range(x.shape[0]):
        for oc in range(out_c):
            ic = oc // multiplier
            for oh in range(grad_out.shape[2]):
                for ow in range(grad_out.shape[3]):
                    grad_val = grad_out[ni, oc, oh, ow]
                    for r in range(kh):
                        for s in range(kw):
                            grad_weight[oc, 0, r, s] += x[ni, ic, oh + r, ow + s] * grad_val
                            grad_input[ni, ic, oh + r, ow + s] += weight[oc, 0, r, s] * grad_val
    return grad_weight, grad_input


def _layernorm2d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=1, keepdims=True).astype(np.float32)
    var = x.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    out = x_hat * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
    return out.astype(np.float32), x_hat, inv_std, mean


def _layernorm2d_backward(
    grad_out: np.ndarray,
    x_hat: np.ndarray,
    inv_std: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_weight = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_bias = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = grad_out * weight.reshape(1, -1, 1, 1)
    channels = float(x_hat.shape[1])
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_input = (inv_std / channels) * (channels * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    return grad_input.astype(np.float32), grad_weight, grad_bias


def _pointwise_forward(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.einsum("nchw,oc->nohw", x, weight[:, :, 0, 0], optimize=True).astype(np.float32)


def _pointwise_backward(grad_out: np.ndarray, x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_weight = np.einsum("nohw,nchw->oc", grad_out, x, optimize=True).astype(np.float32)[:, :, None, None]
    grad_input = np.einsum("nohw,oc->nchw", grad_out, weight[:, :, 0, 0], optimize=True).astype(np.float32)
    return grad_weight, grad_input


def _reference_step(
    x: np.ndarray,
    labels: np.ndarray,
    depthwise_weight: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    pointwise1_weight: np.ndarray,
    pointwise2_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    lr: float,
    eps: float,
) -> dict[str, np.ndarray | float]:
    depthwise = _depthwise_valid(x, depthwise_weight)
    norm, x_hat, inv_std, _mean = _layernorm2d_forward(depthwise, norm_weight, norm_bias, eps)
    pointwise1 = _pointwise_forward(norm, pointwise1_weight)
    activation = _gelu(pointwise1)
    pointwise2 = _pointwise_forward(activation, pointwise2_weight)
    flat = pointwise2.reshape(x.shape[0], -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pointwise2 = (grad_logits @ linear_weight).reshape(pointwise2.shape)
    grad_pointwise2_weight, grad_activation = _pointwise_backward(grad_pointwise2, activation, pointwise2_weight)
    grad_pointwise1 = grad_activation * _gelu_grad(pointwise1)
    grad_pointwise1_weight, grad_norm = _pointwise_backward(grad_pointwise1, norm, pointwise1_weight)
    grad_depthwise, grad_norm_weight, grad_norm_bias = _layernorm2d_backward(grad_norm, x_hat, inv_std, norm_weight)
    grad_depthwise_weight, _grad_input = _depthwise_backward(grad_depthwise, x, depthwise_weight)
    loss_sum = float(-np.log(probs[np.arange(labels.shape[0]), labels] + 1e-10).sum())
    return {
        "depthwise_weight": depthwise_weight - lr * grad_depthwise_weight,
        "norm_weight": norm_weight - lr * grad_norm_weight,
        "norm_bias": norm_bias - lr * grad_norm_bias,
        "pointwise1_weight": pointwise1_weight - lr * grad_pointwise1_weight,
        "pointwise2_weight": pointwise2_weight - lr * grad_pointwise2_weight,
        "linear_weight": linear_weight - lr * grad_linear_weight,
        "linear_bias": linear_bias - lr * grad_linear_bias,
        "loss_mean": loss_sum / float(labels.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cuda_native native GPU ConvNeXt-style bridge training step on a CIFAR-10 batch."
    )
    parser.add_argument("--data-root", default="data/cifar-10-batches-py")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=6)
    parser.add_argument("--allow-random-fallback", action="store_true")
    args = parser.parse_args()

    batch_size = max(1, int(args.batch_size))
    x, labels, dataset_source = _load_batch(
        args.data_root,
        batch_size,
        allow_random_fallback=bool(args.allow_random_fallback),
    )
    rng = np.random.default_rng(int(args.seed))
    in_channels = int(x.shape[1])
    hidden_channels = max(1, int(args.hidden_channels))
    depthwise_weight = rng.normal(0.0, 0.03, size=(in_channels, 1, 3, 3)).astype(np.float32)
    norm_weight = np.ones((in_channels,), dtype=np.float32)
    norm_bias = np.zeros((in_channels,), dtype=np.float32)
    pointwise1_weight = rng.normal(0.0, 0.03, size=(hidden_channels, in_channels, 1, 1)).astype(np.float32)
    pointwise2_weight = rng.normal(0.0, 0.03, size=(in_channels, hidden_channels, 1, 1)).astype(np.float32)
    flat_features = in_channels * 30 * 30
    linear_weight = rng.normal(0.0, 0.01, size=(10, flat_features)).astype(np.float32)
    linear_bias = np.zeros((10,), dtype=np.float32)

    result = native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise1_weight,
        pointwise2_weight,
        linear_weight,
        linear_bias,
        lr=float(args.lr),
        norm_eps=float(args.eps),
    )
    ref = _reference_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise1_weight,
        pointwise2_weight,
        linear_weight,
        linear_bias,
        float(args.lr),
        float(args.eps),
    )
    payload = {
        "demo": "cuda_native_gpu_convnext_bridge_training_cifar10",
        "dataset_source": dataset_source,
        "batch_shape": list(x.shape),
        "loss_mean": result.loss_mean,
        "reference_loss_mean": ref["loss_mean"],
        "max_abs_updated_depthwise_weight_diff_vs_reference": float(
            np.max(np.abs(result.updated_depthwise_weight - ref["depthwise_weight"]))
        ),
        "max_abs_updated_norm_weight_diff_vs_reference": float(
            np.max(np.abs(result.updated_norm_weight - ref["norm_weight"]))
        ),
        "max_abs_updated_norm_bias_diff_vs_reference": float(
            np.max(np.abs(result.updated_norm_bias - ref["norm_bias"]))
        ),
        "max_abs_updated_pointwise1_weight_diff_vs_reference": float(
            np.max(np.abs(result.updated_pointwise1_weight - ref["pointwise1_weight"]))
        ),
        "max_abs_updated_pointwise2_weight_diff_vs_reference": float(
            np.max(np.abs(result.updated_pointwise2_weight - ref["pointwise2_weight"]))
        ),
        "max_abs_updated_linear_weight_diff_vs_reference": float(
            np.max(np.abs(result.updated_linear_weight - ref["linear_weight"]))
        ),
        "max_abs_updated_linear_bias_diff_vs_reference": float(
            np.max(np.abs(result.updated_linear_bias - ref["linear_bias"]))
        ),
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
