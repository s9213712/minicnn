"""Loss functions for cuda_native training.

Each function returns (scalar_loss, grad_logits) so the caller can
feed grad_logits directly into the backward executor.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Softmax cross-entropy for integer class labels.

    Args:
        logits: (N, C) float32
        labels: (N,) int  class indices in [0, C)

    Returns:
        (loss, grad_logits) where loss is a Python float and
        grad_logits is (N, C) float32.
    """
    n = logits.shape[0]
    probs = softmax(logits)
    log_probs = np.log(probs.clip(min=1e-12))
    loss = -float(log_probs[np.arange(n), labels.astype(int)].mean())
    grad = probs.copy()
    grad[np.arange(n), labels.astype(int)] -= 1.0
    grad /= n
    return loss, grad.astype(np.float32)


def mse_loss(
    preds: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean squared error: loss = mean((preds - targets)^2).

    Args:
        preds:   (N, ...) float32
        targets: same shape as preds, float32

    Returns:
        (loss, grad_preds)
    """
    diff = preds - targets.astype(np.float32)
    loss = float(np.mean(diff ** 2))
    grad = (2.0 * diff / diff.size).astype(np.float32)
    return loss, grad
