"""Host-side model weight initialization shared by CUDA and PyTorch trainers."""

import numpy as np

from minicnn.config.settings import C1_IN, C1_OUT, C2_IN, C2_OUT, C3_IN, C3_OUT, C4_IN, C4_OUT, FC_IN, KH, KW


def he_init(size, fan_in, rng=None):
    rng = rng or np.random.default_rng()
    return (rng.standard_normal(size).astype(np.float32) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def init_weights(seed):
    rng = np.random.default_rng(seed)
    w_conv1 = he_init(C1_OUT * C1_IN * KH * KW, C1_IN * KH * KW, rng)
    w_conv2 = he_init(C2_OUT * C2_IN * KH * KW, C2_IN * KH * KW, rng)
    w_conv3 = he_init(C3_OUT * C3_IN * KH * KW, C3_IN * KH * KW, rng)
    w_conv4 = he_init(C4_OUT * C4_IN * KH * KW, C4_IN * KH * KW, rng)
    fc_w = he_init(10 * FC_IN, FC_IN, rng)
    fc_b = np.zeros(10, dtype=np.float32)
    return w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b
