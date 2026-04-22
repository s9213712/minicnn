"""GPU pointer container types for handcrafted CUDA training."""
from __future__ import annotations

from typing import Iterator


class DeviceWeights:
    """GPU pointers for all trainable weights."""

    def __init__(self, conv_weights: list, fc_w, fc_b,
                 ln_gamma: list | None = None, ln_beta: list | None = None,
                 bn_gamma: list | None = None, bn_beta: list | None = None,
                 bn_running_mean: list | None = None, bn_running_var: list | None = None) -> None:
        self.conv_weights = list(conv_weights)
        self.fc_w = fc_w
        self.fc_b = fc_b
        self.ln_gamma: list = list(ln_gamma) if ln_gamma else []
        self.ln_beta: list = list(ln_beta) if ln_beta else []
        self.bn_gamma: list = list(bn_gamma) if bn_gamma else []
        self.bn_beta: list = list(bn_beta) if bn_beta else []
        self.bn_running_mean: list = list(bn_running_mean) if bn_running_mean else []
        self.bn_running_var: list = list(bn_running_var) if bn_running_var else []

    def __iter__(self) -> Iterator:
        yield from self.conv_weights
        yield self.fc_w
        yield self.fc_b
        yield from self.ln_gamma
        yield from self.ln_beta
        yield from self.bn_gamma
        yield from self.bn_beta
        yield from self.bn_running_mean
        yield from self.bn_running_var


class VelocityBuffers:
    """GPU momentum velocity buffers matching DeviceWeights layout."""

    def __init__(self, conv_velocities: list, fc_w_vel, fc_b_vel,
                 ln_gamma_vel: list | None = None, ln_beta_vel: list | None = None,
                 bn_gamma_vel: list | None = None, bn_beta_vel: list | None = None) -> None:
        self.conv_velocities = list(conv_velocities)
        self.fc_w_vel = fc_w_vel
        self.fc_b_vel = fc_b_vel
        self.ln_gamma_vel: list = list(ln_gamma_vel) if ln_gamma_vel else []
        self.ln_beta_vel: list = list(ln_beta_vel) if ln_beta_vel else []
        self.bn_gamma_vel: list = list(bn_gamma_vel) if bn_gamma_vel else []
        self.bn_beta_vel: list = list(bn_beta_vel) if bn_beta_vel else []

    def __iter__(self) -> Iterator:
        yield from self.conv_velocities
        yield self.fc_w_vel
        yield self.fc_b_vel
        yield from self.ln_gamma_vel
        yield from self.ln_beta_vel
        yield from self.bn_gamma_vel
        yield from self.bn_beta_vel


class AdamBuffers:
    """GPU first/second moment buffers for Adam, mirroring DeviceWeights layout."""

    def __init__(self, conv_m: list, conv_v: list, fc_w_m, fc_w_v, fc_b_m, fc_b_v,
                 bn_gamma_m: list | None = None, bn_gamma_v: list | None = None,
                 bn_beta_m: list | None = None, bn_beta_v: list | None = None) -> None:
        self.conv_m = list(conv_m)
        self.conv_v = list(conv_v)
        self.fc_w_m = fc_w_m
        self.fc_w_v = fc_w_v
        self.fc_b_m = fc_b_m
        self.fc_b_v = fc_b_v
        self.bn_gamma_m: list = list(bn_gamma_m) if bn_gamma_m else []
        self.bn_gamma_v: list = list(bn_gamma_v) if bn_gamma_v else []
        self.bn_beta_m: list = list(bn_beta_m) if bn_beta_m else []
        self.bn_beta_v: list = list(bn_beta_v) if bn_beta_v else []

    def __iter__(self):
        yield from self.conv_m
        yield from self.conv_v
        yield self.fc_w_m
        yield self.fc_w_v
        yield self.fc_b_m
        yield self.fc_b_v
        yield from self.bn_gamma_m
        yield from self.bn_gamma_v
        yield from self.bn_beta_m
        yield from self.bn_beta_v
