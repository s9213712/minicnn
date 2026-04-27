from __future__ import annotations

import random as py_random

import numpy as np

_GLOBAL_RNG = np.random.default_rng()


def get_global_rng() -> np.random.Generator:
    return _GLOBAL_RNG


def set_global_seed(seed: int) -> np.random.Generator:
    global _GLOBAL_RNG
    seed = int(seed)
    py_random.seed(seed)
    np.random.seed(seed)
    _GLOBAL_RNG = np.random.default_rng(seed)
    return _GLOBAL_RNG
