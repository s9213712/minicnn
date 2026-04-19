from __future__ import annotations

import numpy as np


class MemoryPool:
    def __init__(self):
        self._free: dict[tuple[tuple[int, ...], str], list[np.ndarray]] = {}

    def alloc(self, shape, dtype=np.float32):
        key = (tuple(shape), np.dtype(dtype).str)
        bucket = self._free.get(key)
        if bucket:
            return bucket.pop()
        return np.empty(shape, dtype=dtype)

    def release(self, array) -> None:
        key = (tuple(array.shape), array.dtype.str)
        self._free.setdefault(key, []).append(array)
