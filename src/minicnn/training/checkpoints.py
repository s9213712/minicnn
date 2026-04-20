"""Weight checkpointing and device pointer management."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from minicnn.core.cuda_backend import g2h, gpu_zeros, lib, upload
from minicnn.training.cuda_arch import CudaNetGeometry


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
        self.ln_beta: list  = list(ln_beta)  if ln_beta  else []
        self.bn_gamma: list = list(bn_gamma) if bn_gamma else []
        self.bn_beta: list  = list(bn_beta)  if bn_beta  else []
        self.bn_running_mean: list = list(bn_running_mean) if bn_running_mean else []
        self.bn_running_var: list  = list(bn_running_var)  if bn_running_var  else []

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
        self.ln_beta_vel: list  = list(ln_beta_vel)  if ln_beta_vel  else []
        self.bn_gamma_vel: list = list(bn_gamma_vel) if bn_gamma_vel else []
        self.bn_beta_vel: list  = list(bn_beta_vel)  if bn_beta_vel  else []

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
        self.bn_beta_m: list  = list(bn_beta_m)  if bn_beta_m  else []
        self.bn_beta_v: list  = list(bn_beta_v)  if bn_beta_v  else []

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


def init_adam_buffers(geom: CudaNetGeometry) -> AdamBuffers:
    bn_stages = [s for s in geom.conv_stages if s.batch_norm]
    return AdamBuffers(
        conv_m=[gpu_zeros(s.weight_numel) for s in geom.conv_stages],
        conv_v=[gpu_zeros(s.weight_numel) for s in geom.conv_stages],
        fc_w_m=gpu_zeros(geom.fc_out * geom.fc_in),
        fc_w_v=gpu_zeros(geom.fc_out * geom.fc_in),
        fc_b_m=gpu_zeros(geom.fc_out),
        fc_b_v=gpu_zeros(geom.fc_out),
        bn_gamma_m=[gpu_zeros(s.out_c) for s in bn_stages],
        bn_gamma_v=[gpu_zeros(s.out_c) for s in bn_stages],
        bn_beta_m=[gpu_zeros(s.out_c) for s in bn_stages],
        bn_beta_v=[gpu_zeros(s.out_c) for s in bn_stages],
    )


def upload_weights(
    conv_arrays: list[np.ndarray],
    fc_w: np.ndarray,
    fc_b: np.ndarray,
    ln_gamma_arrays: list[np.ndarray] | None = None,
    ln_beta_arrays: list[np.ndarray] | None = None,
    bn_gamma_arrays: list[np.ndarray] | None = None,
    bn_beta_arrays: list[np.ndarray] | None = None,
    bn_running_mean_arrays: list[np.ndarray] | None = None,
    bn_running_var_arrays: list[np.ndarray] | None = None,
) -> DeviceWeights:
    uploaded: list = []
    ln_gamma_arrays        = ln_gamma_arrays        or []
    ln_beta_arrays         = ln_beta_arrays         or []
    bn_gamma_arrays        = bn_gamma_arrays        or []
    bn_beta_arrays         = bn_beta_arrays         or []
    bn_running_mean_arrays = bn_running_mean_arrays or []
    bn_running_var_arrays  = bn_running_var_arrays  or []
    all_arrays = (
        *conv_arrays, fc_w, fc_b,
        *ln_gamma_arrays, *ln_beta_arrays,
        *bn_gamma_arrays, *bn_beta_arrays,
        *bn_running_mean_arrays, *bn_running_var_arrays,
    )
    try:
        for arr in all_arrays:
            uploaded.append(upload(arr))
        n_conv  = len(conv_arrays)
        n_ln    = len(ln_gamma_arrays)
        n_bn    = len(bn_gamma_arrays)
        base    = n_conv + 2
        return DeviceWeights(
            conv_weights=uploaded[:n_conv],
            fc_w=uploaded[n_conv],
            fc_b=uploaded[n_conv + 1],
            ln_gamma=uploaded[base : base + n_ln],
            ln_beta=uploaded[base + n_ln : base + 2 * n_ln],
            bn_gamma=uploaded[base + 2 * n_ln : base + 2 * n_ln + n_bn],
            bn_beta=uploaded[base + 2 * n_ln + n_bn : base + 2 * n_ln + 2 * n_bn],
            bn_running_mean=uploaded[base + 2 * n_ln + 2 * n_bn : base + 2 * n_ln + 3 * n_bn],
            bn_running_var=uploaded[base + 2 * n_ln + 3 * n_bn :],
        )
    except Exception:
        free_weights(uploaded)
        raise


def init_velocity_buffers(geom: CudaNetGeometry) -> VelocityBuffers:
    ln_stages = [s for s in geom.conv_stages if s.layer_norm]
    bn_stages = [s for s in geom.conv_stages if s.batch_norm]
    return VelocityBuffers(
        conv_velocities=[gpu_zeros(s.weight_numel) for s in geom.conv_stages],
        fc_w_vel=gpu_zeros(geom.fc_out * geom.fc_in),
        fc_b_vel=gpu_zeros(geom.fc_out),
        ln_gamma_vel=[gpu_zeros(s.out_c) for s in ln_stages],
        ln_beta_vel=[gpu_zeros(s.out_c) for s in ln_stages],
        bn_gamma_vel=[gpu_zeros(s.out_c) for s in bn_stages],
        bn_beta_vel=[gpu_zeros(s.out_c) for s in bn_stages],
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
    bn_stages = [(i, s) for i, s in enumerate(geom.conv_stages) if s.batch_norm]
    bn_idx = 0
    bn_data: dict = {}
    for i, s in bn_stages:
        bn_data[f'bn_gamma{i + 1}']       = g2h(device_weights.bn_gamma[bn_idx], s.out_c)
        bn_data[f'bn_beta{i + 1}']        = g2h(device_weights.bn_beta[bn_idx], s.out_c)
        bn_data[f'bn_running_mean{i + 1}'] = g2h(device_weights.bn_running_mean[bn_idx], s.out_c)
        bn_data[f'bn_running_var{i + 1}']  = g2h(device_weights.bn_running_var[bn_idx], s.out_c)
        bn_idx += 1
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
        **bn_data,
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
    bn_stages = [(i, s) for i, s in enumerate(geom.conv_stages) if s.batch_norm]
    bn_gamma_arrays, bn_beta_arrays, bn_rm_arrays, bn_rv_arrays = [], [], [], []
    for i, s in bn_stages:
        key_g = f'bn_gamma{i + 1}'
        key_b = f'bn_beta{i + 1}'
        key_m = f'bn_running_mean{i + 1}'
        key_v = f'bn_running_var{i + 1}'
        bn_gamma_arrays.append(ckpt[key_g].astype(np.float32) if key_g in ckpt else np.ones(s.out_c, dtype=np.float32))
        bn_beta_arrays.append(ckpt[key_b].astype(np.float32) if key_b in ckpt else np.zeros(s.out_c, dtype=np.float32))
        bn_rm_arrays.append(ckpt[key_m].astype(np.float32) if key_m in ckpt else np.zeros(s.out_c, dtype=np.float32))
        bn_rv_arrays.append(ckpt[key_v].astype(np.float32) if key_v in ckpt else np.ones(s.out_c, dtype=np.float32))
    uploaded: list = []
    try:
        for arr in (*conv_arrays, fc_w, fc_b,
                    *bn_gamma_arrays, *bn_beta_arrays, *bn_rm_arrays, *bn_rv_arrays):
            uploaded.append(upload(arr))
        n_bn = len(bn_stages)
        new_dw = DeviceWeights(
            conv_weights=uploaded[: geom.n_conv],
            fc_w=uploaded[geom.n_conv],
            fc_b=uploaded[geom.n_conv + 1],
            bn_gamma=uploaded[geom.n_conv + 2 : geom.n_conv + 2 + n_bn],
            bn_beta=uploaded[geom.n_conv + 2 + n_bn : geom.n_conv + 2 + 2 * n_bn],
            bn_running_mean=uploaded[geom.n_conv + 2 + 2 * n_bn : geom.n_conv + 2 + 3 * n_bn],
            bn_running_var=uploaded[geom.n_conv + 2 + 3 * n_bn :],
        )
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
