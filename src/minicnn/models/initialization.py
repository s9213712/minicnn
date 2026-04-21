from __future__ import annotations

import math

import numpy as np


def _fans(shape):
    if len(shape) == 2:
        fan_in = shape[1]
        fan_out = shape[0]
        return fan_in, fan_out
    fan_in = shape[1] * shape[2] * shape[3]
    fan_out = shape[0] * shape[2] * shape[3]
    return fan_in, fan_out


def kaiming_uniform(shape, a: float = 0, mode: str = 'fan_in', rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    fan_in, fan_out = _fans(shape)
    fan = fan_in if mode == 'fan_in' else fan_out
    bound = math.sqrt(6.0 / ((1 + a * a) * max(fan, 1)))
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def kaiming_normal(shape, a: float = 0, mode: str = 'fan_in', rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    fan_in, fan_out = _fans(shape)
    fan = fan_in if mode == 'fan_in' else fan_out
    std = math.sqrt(2.0 / ((1 + a * a) * max(fan, 1)))
    return (rng.standard_normal(shape) * std).astype(np.float32)


def xavier_uniform(shape, gain: float = 1.0, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    fan_in, fan_out = _fans(shape)
    bound = gain * math.sqrt(6.0 / max(fan_in + fan_out, 1))
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def xavier_normal(shape, gain: float = 1.0, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    fan_in, fan_out = _fans(shape)
    std = gain * math.sqrt(2.0 / max(fan_in + fan_out, 1))
    return (rng.standard_normal(shape) * std).astype(np.float32)


def normal_init(shape, mean: float = 0.0, std: float = 0.01, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (rng.standard_normal(shape) * std + mean).astype(np.float32)


def zeros_init(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def get_initializer(name: str):
    _MAP = {
        'kaiming_uniform': kaiming_uniform,
        'kaiming_normal': kaiming_normal,
        'xavier_uniform': xavier_uniform,
        'xavier_normal': xavier_normal,
        'normal': normal_init,
        'zeros': zeros_init,
        'he': kaiming_uniform,
    }
    try:
        return _MAP[name]
    except KeyError as exc:
        raise KeyError(f'Unknown initializer {name!r}; expected one of: {", ".join(sorted(_MAP))}') from exc


def he_init(size: int, fan_in: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (rng.standard_normal(size) * np.sqrt(2.0 / fan_in)).astype(np.float32)


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
