"""Weight checkpointing and device pointer management."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from minicnn.core.cuda_backend import g2h, gpu_zeros, lib, upload
from minicnn.training.cuda_arch import CudaNetGeometry


class DeviceWeights:
    """GPU pointers for all trainable weights."""

    def __init__(self, conv_weights: list, fc_w, fc_b) -> None:
        self.conv_weights = list(conv_weights)
        self.fc_w = fc_w
        self.fc_b = fc_b

    def __iter__(self) -> Iterator:
        yield from self.conv_weights
        yield self.fc_w
        yield self.fc_b


class VelocityBuffers:
    """GPU momentum velocity buffers matching DeviceWeights layout."""

    def __init__(self, conv_velocities: list, fc_w_vel, fc_b_vel) -> None:
        self.conv_velocities = list(conv_velocities)
        self.fc_w_vel = fc_w_vel
        self.fc_b_vel = fc_b_vel

    def __iter__(self) -> Iterator:
        yield from self.conv_velocities
        yield self.fc_w_vel
        yield self.fc_b_vel


def upload_weights(conv_arrays: list[np.ndarray], fc_w: np.ndarray, fc_b: np.ndarray) -> DeviceWeights:
    return DeviceWeights(
        conv_weights=[upload(w) for w in conv_arrays],
        fc_w=upload(fc_w),
        fc_b=upload(fc_b),
    )


def init_velocity_buffers(geom: CudaNetGeometry) -> VelocityBuffers:
    return VelocityBuffers(
        conv_velocities=[gpu_zeros(s.weight_numel) for s in geom.conv_stages],
        fc_w_vel=gpu_zeros(geom.fc_out * geom.fc_in),
        fc_b_vel=gpu_zeros(geom.fc_out),
    )


def save_checkpoint(
    path: str,
    epoch: int,
    val_acc: float,
    lr_conv1: float,
    lr_conv: float,
    lr_fc: float,
    device_weights: DeviceWeights,
    geom: CudaNetGeometry,
) -> None:
    conv_data = {
        f'w_conv{i + 1}': g2h(dw, s.weight_numel)
        for i, (dw, s) in enumerate(zip(device_weights.conv_weights, geom.conv_stages))
    }
    np.savez(
        path,
        epoch=np.int32(epoch),
        val_acc=np.float32(val_acc),
        lr_conv1=np.float32(lr_conv1),
        lr_conv=np.float32(lr_conv),
        lr_fc=np.float32(lr_fc),
        n_conv=np.int32(geom.n_conv),
        fc_w=g2h(device_weights.fc_w, geom.fc_out * geom.fc_in),
        fc_b=g2h(device_weights.fc_b, geom.fc_out),
        **conv_data,
    )


def reload_weights_from_checkpoint(
    path: str,
    device_weights: DeviceWeights,
    geom: CudaNetGeometry,
) -> tuple:
    ckpt = np.load(path)
    saved_n_conv = int(ckpt['n_conv'])
    if saved_n_conv != geom.n_conv:
        raise ValueError(
            f"Checkpoint '{path}' has n_conv={saved_n_conv} but current architecture has n_conv={geom.n_conv}"
        )
    conv_arrays = [ckpt[f'w_conv{i + 1}'].astype(np.float32) for i in range(geom.n_conv)]
    fc_w = ckpt['fc_w'].astype(np.float32)
    fc_b = ckpt['fc_b'].astype(np.float32)
    uploaded: list = []
    try:
        for arr in (*conv_arrays, fc_w, fc_b):
            uploaded.append(upload(arr))
        new_dw = DeviceWeights(uploaded[: geom.n_conv], uploaded[-2], uploaded[-1])
    except Exception:
        free_weights(uploaded)
        raise
    free_weights(device_weights)
    return ckpt, fc_w, fc_b, new_dw


def free_weights(device_weights) -> None:
    """Free all GPU pointers in device_weights (accepts DeviceWeights or a plain list)."""
    if device_weights is None:
        return
    for ptr in device_weights:
        if ptr is not None:
            lib.gpu_free(ptr)
