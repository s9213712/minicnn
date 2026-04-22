from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from minicnn.checkpoint_schema import CHECKPOINT_SCHEMA_VERSION, CUDA_LEGACY_CHECKPOINT_KIND
from minicnn.training.cuda_arch import CudaNetGeometry


def legacy_checkpoint_path(path: str | Path) -> tuple[Path, Path]:
    path_obj = Path(path)
    if path_obj.suffix != '.npz':
        path_obj = path_obj.with_suffix('.npz')
    return path_obj, path_obj.with_suffix('.tmp.npz')


def build_legacy_checkpoint_payload(
    *,
    epoch: int,
    val_acc: float,
    lr_conv1: float,
    lr_conv: float,
    lr_fc: float,
    device_weights,
    geom: CudaNetGeometry,
    g2h_fn,
) -> dict[str, Any]:
    conv_data = {
        f'w_conv{i + 1}': g2h_fn(dw, s.weight_numel)
        for i, (dw, s) in enumerate(zip(device_weights.conv_weights, geom.conv_stages))
    }
    bn_stages = [(i, s) for i, s in enumerate(geom.conv_stages) if s.batch_norm]
    bn_payload: dict[str, Any] = {}
    for bn_idx, (i, s) in enumerate(bn_stages):
        bn_payload[f'bn_gamma{i + 1}'] = g2h_fn(device_weights.bn_gamma[bn_idx], s.out_c)
        bn_payload[f'bn_beta{i + 1}'] = g2h_fn(device_weights.bn_beta[bn_idx], s.out_c)
        bn_payload[f'bn_running_mean{i + 1}'] = g2h_fn(device_weights.bn_running_mean[bn_idx], s.out_c)
        bn_payload[f'bn_running_var{i + 1}'] = g2h_fn(device_weights.bn_running_var[bn_idx], s.out_c)
    return {
        'schema_version': np.int32(CHECKPOINT_SCHEMA_VERSION),
        'backend': np.str_('cuda_legacy'),
        'checkpoint_kind': np.str_(CUDA_LEGACY_CHECKPOINT_KIND),
        'created_at': np.str_(datetime.now(timezone.utc).isoformat()),
        'epoch': np.int32(epoch),
        'val_acc': np.float32(val_acc),
        'lr_conv1': np.float32(lr_conv1),
        'lr_conv': np.float32(lr_conv),
        'lr_fc': np.float32(lr_fc),
        'n_conv': np.int32(geom.n_conv),
        'fc_w': g2h_fn(device_weights.fc_w, geom.fc_out * geom.fc_in),
        'fc_b': g2h_fn(device_weights.fc_b, geom.fc_out),
        **conv_data,
        **bn_payload,
    }


def load_legacy_checkpoint_arrays(path: str | Path, geom: CudaNetGeometry):
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
    return ckpt, conv_arrays, fc_w, fc_b, bn_stages, bn_gamma_arrays, bn_beta_arrays, bn_rm_arrays, bn_rv_arrays
