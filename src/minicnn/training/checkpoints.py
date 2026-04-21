"""Weight checkpointing and device pointer cleanup."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from minicnn.core.cuda_backend import g2h, gpu_zeros, lib, upload
from minicnn.config.settings import C1_IN, C1_OUT, C2_IN, C2_OUT, C3_IN, C3_OUT, C4_IN, C4_OUT, FC_IN, KH, KW


@dataclass(frozen=True)
class DeviceWeights:
    w_conv1: object
    w_conv2: object
    w_conv3: object
    w_conv4: object
    fc_w: object
    fc_b: object

    def __iter__(self) -> Iterator[object]:
        return iter((self.w_conv1, self.w_conv2, self.w_conv3, self.w_conv4, self.fc_w, self.fc_b))

    def as_tuple(self) -> tuple[object, object, object, object, object, object]:
        return tuple(self)


@dataclass(frozen=True)
class VelocityBuffers:
    v_conv1: object
    v_conv2: object
    v_conv3: object
    v_conv4: object
    v_fc_w: object
    v_fc_b: object

    def __iter__(self) -> Iterator[object]:
        return iter((self.v_conv1, self.v_conv2, self.v_conv3, self.v_conv4, self.v_fc_w, self.v_fc_b))

    def as_tuple(self) -> tuple[object, object, object, object, object, object]:
        return tuple(self)


def upload_weights(w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b):
    return DeviceWeights(
        w_conv1=upload(w_conv1),
        w_conv2=upload(w_conv2),
        w_conv3=upload(w_conv3),
        w_conv4=upload(w_conv4),
        fc_w=upload(fc_w),
        fc_b=upload(fc_b),
    )


def init_velocity_buffers():
    return VelocityBuffers(
        v_conv1=gpu_zeros(C1_OUT * C1_IN * KH * KW),
        v_conv2=gpu_zeros(C2_OUT * C2_IN * KH * KW),
        v_conv3=gpu_zeros(C3_OUT * C3_IN * KH * KW),
        v_conv4=gpu_zeros(C4_OUT * C4_IN * KH * KW),
        v_fc_w=gpu_zeros(10 * FC_IN),
        v_fc_b=gpu_zeros(10),
    )


def save_checkpoint(path, epoch, val_acc, lr_conv1, lr_conv, lr_fc, device_weights):
    path = Path(path)
    if path.suffix != '.npz':
        path = path.with_suffix('.npz')
    tmp = path.with_suffix('.tmp.npz')
    d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = device_weights
    try:
        np.savez(
            str(tmp),
            epoch=np.int32(epoch),
            val_acc=np.float32(val_acc),
            lr_conv1=np.float32(lr_conv1),
            lr_conv=np.float32(lr_conv),
            lr_fc=np.float32(lr_fc),
            w_conv1=g2h(d_w_conv1, C1_OUT * C1_IN * KH * KW),
            w_conv2=g2h(d_w_conv2, C2_OUT * C2_IN * KH * KW),
            w_conv3=g2h(d_w_conv3, C3_OUT * C3_IN * KH * KW),
            w_conv4=g2h(d_w_conv4, C4_OUT * C4_IN * KH * KW),
            fc_w=g2h(d_fc_w, 10 * FC_IN),
            fc_b=g2h(d_fc_b, 10),
        )
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def reload_weights_from_checkpoint(path, device_weights):
    ckpt = np.load(path)
    fc_w = ckpt["fc_w"].astype(np.float32)
    fc_b = ckpt["fc_b"].astype(np.float32)
    uploaded: list[object] = []
    try:
        for array in (
            ckpt["w_conv1"].astype(np.float32),
            ckpt["w_conv2"].astype(np.float32),
            ckpt["w_conv3"].astype(np.float32),
            ckpt["w_conv4"].astype(np.float32),
            fc_w,
            fc_b,
        ):
            uploaded.append(upload(array))
        new_device_weights = DeviceWeights(*uploaded)
    except Exception:
        free_weights(uploaded)
        raise
    free_weights(device_weights)
    return ckpt, fc_w, fc_b, new_device_weights


def free_weights(device_weights):
    for ptr in device_weights:
        lib.gpu_free(ptr)
