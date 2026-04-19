"""Epoch-level helpers for the legacy CUDA trainer."""

from __future__ import annotations

import numpy as np


def random_crop_batch(x: np.ndarray, rng: np.random.Generator, padding: int) -> np.ndarray:
    if padding <= 0:
        return x
    n, c, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
    tops = rng.integers(0, 2 * padding + 1, size=n)
    lefts = rng.integers(0, 2 * padding + 1, size=n)
    batch_idx = np.arange(n)[:, None, None, None]
    channel_idx = np.arange(c)[None, :, None, None]
    row_idx = tops[:, None, None, None] + np.arange(h)[None, None, :, None]
    col_idx = lefts[:, None, None, None] + np.arange(w)[None, None, None, :]
    cropped = padded[batch_idx, channel_idx, row_idx, col_idx]
    return np.ascontiguousarray(cropped.astype(np.float32, copy=False))


def augment_batch(
    x: np.ndarray,
    rng: np.random.Generator,
    random_crop_padding: int,
    horizontal_flip: bool,
) -> np.ndarray:
    """Always returns a new array; does not alias the input."""
    x = random_crop_batch(x, rng, random_crop_padding)
    x = x.copy()
    if horizontal_flip:
        flip_mask = rng.random(x.shape[0]) > 0.5
        if flip_mask.any():
            x[flip_mask] = x[flip_mask, :, :, ::-1]
    return x
