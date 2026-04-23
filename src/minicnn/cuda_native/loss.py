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
    *,
    label_smoothing: float = 0.0,
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
    labels_i = labels.astype(int)
    if label_smoothing > 0.0:
        num_classes = logits.shape[1]
        targets = np.full((n, num_classes), label_smoothing / num_classes, dtype=np.float32)
        targets[np.arange(n), labels_i] += 1.0 - label_smoothing
        loss = -float(np.sum(targets * log_probs) / n)
        grad = probs - targets
    else:
        loss = -float(log_probs[np.arange(n), labels_i].mean())
        grad = probs.copy()
        grad[np.arange(n), labels_i] -= 1.0
    grad /= n
    return loss, grad.astype(np.float32)


def bce_with_logits_loss(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits_f = np.asarray(logits, dtype=np.float32)
    if logits_f.ndim == 2 and logits_f.shape[1] == 1:
        logits_f = logits_f.reshape(-1)
    targets_f = np.asarray(targets, dtype=np.float32).reshape(-1)
    if logits_f.ndim != 1:
        raise ValueError(
            f'BCEWithLogitsLoss expects logits shaped (N,) or (N,1), got {logits.shape}'
        )
    if logits_f.shape[0] != targets_f.shape[0]:
        raise ValueError(
            f'BCEWithLogitsLoss expects the same batch size for logits and targets, got {logits_f.shape[0]} and {targets_f.shape[0]}'
        )
    if not np.all((targets_f == 0.0) | (targets_f == 1.0)):
        raise ValueError('BCEWithLogitsLoss expects binary targets in {0, 1}.')
    max_term = np.maximum(logits_f, 0.0)
    loss = float(np.mean(max_term - logits_f * targets_f + np.log1p(np.exp(-np.abs(logits_f)))))
    probs = (1.0 / (1.0 + np.exp(-logits_f))).astype(np.float32)
    grad = ((probs - targets_f) / max(logits_f.shape[0], 1)).astype(np.float32)
    return loss, grad.reshape(logits.shape).astype(np.float32)


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
