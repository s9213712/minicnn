"""Weight checkpointing and device pointer management."""
from __future__ import annotations

import os

import numpy as np

from minicnn.core.cuda_backend import g2h, gpu_zeros, lib, upload
from minicnn.training.cuda_arch import CudaNetGeometry
from minicnn.training._checkpoint_payloads import (
    build_legacy_checkpoint_payload,
    legacy_checkpoint_path,
    load_legacy_checkpoint_arrays,
)
from minicnn.training._weight_buffers import AdamBuffers, DeviceWeights, VelocityBuffers


def _upload_arrays_transactionally(arrays: tuple[np.ndarray, ...] | list[np.ndarray]) -> list:
    uploaded: list = []
    try:
        for arr in arrays:
            uploaded.append(upload(arr))
        return uploaded
    except Exception:
        free_weights(uploaded)
        raise


def _save_npz_transactionally(path_obj, tmp, payload: dict[str, np.ndarray | float | int | str]) -> None:
    try:
        np.savez(str(tmp), **payload)
        os.replace(str(tmp), str(path_obj))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


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
    uploaded = _upload_arrays_transactionally(all_arrays)
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
    path_obj, tmp = legacy_checkpoint_path(path)
    payload = build_legacy_checkpoint_payload(
        epoch=epoch,
        val_acc=val_acc,
        lr_conv1=lr_conv1,
        lr_conv=lr_conv,
        lr_fc=lr_fc,
        device_weights=device_weights,
        geom=geom,
        g2h_fn=g2h,
    )
    _save_npz_transactionally(path_obj, tmp, payload)


def reload_weights_from_checkpoint(
    path: str,
    device_weights: DeviceWeights,
    geom: CudaNetGeometry,
) -> tuple:
    ckpt, conv_arrays, fc_w, fc_b, bn_stages, bn_gamma_arrays, bn_beta_arrays, bn_rm_arrays, bn_rv_arrays = load_legacy_checkpoint_arrays(path, geom)
    uploaded = _upload_arrays_transactionally(
        (
            *conv_arrays, fc_w, fc_b,
            *bn_gamma_arrays, *bn_beta_arrays, *bn_rm_arrays, *bn_rv_arrays,
        )
    )
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
    free_weights(device_weights)
    return ckpt, fc_w, fc_b, new_dw


def free_weights(device_weights) -> None:
    """Free all GPU pointers in device_weights (accepts DeviceWeights or a plain list)."""
    if device_weights is None:
        return
    for ptr in device_weights:
        if ptr is not None:
            lib.gpu_free(ptr)
