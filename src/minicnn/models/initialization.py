"""Host-side weight initialisation for the CUDA and PyTorch trainers."""
from __future__ import annotations

import numpy as np


def he_init(size: int, fan_in: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (rng.standard_normal(size).astype(np.float32) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def init_weights(seed: int, geom=None):
    """Return ``(*conv_arrays, fc_w, fc_b)`` initialised with He init.

    ``geom`` is a :class:`CudaNetGeometry`.  When omitted the geometry is read
    from the current global config via :func:`~minicnn.config.settings.get_arch`.

    The return value is a flat tuple so callers can unpack with starred assignment::

        *conv_arrs, fc_w, fc_b = init_weights(seed)

    For a fixed 4-stage architecture the traditional 6-value unpack also works::

        w1, w2, w3, w4, fc_w, fc_b = init_weights(seed)
    """
    if geom is None:
        from minicnn.config.settings import get_arch
        geom = get_arch()
    rng = np.random.default_rng(seed)
    conv_arrays = [
        he_init(s.weight_numel, s.in_c * s.kh * s.kw, rng)
        for s in geom.conv_stages
    ]
    fc_w = he_init(geom.fc_out * geom.fc_in, geom.fc_in, rng)
    fc_b = np.zeros(geom.fc_out, dtype=np.float32)
    return (*conv_arrays, fc_w, fc_b)
