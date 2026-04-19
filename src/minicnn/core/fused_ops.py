from __future__ import annotations

import numpy as np


def compute_bn_affine(running_mean, running_var, gamma, beta, eps: float = 1e-5):
    running_mean = np.asarray(running_mean, dtype=np.float32)
    running_var = np.asarray(running_var, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    if not (running_mean.shape == running_var.shape == gamma.shape == beta.shape):
        raise ValueError('BatchNorm affine inputs must have identical channel shapes')
    scale = gamma / np.sqrt(running_var + eps)
    shift = beta - running_mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)


def can_use_fused_conv_bn_relu(*, kernel_size: int = 3, stride: int = 1, groups: int = 1, dtype=np.float32) -> bool:
    return kernel_size == 3 and stride == 1 and groups == 1 and np.dtype(dtype) == np.dtype(np.float32)


def _conv2d_nchw(x, weight, bias=None, padding: int = 0):
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    out_h = h + 2 * padding - kh + 1
    out_w = w + 2 * padding - kw + 1
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = x_pad[:, :, i:i + kh, j:j + kw]
            out[:, :, i, j] = np.tensordot(region, weight, axes=([1, 2, 3], [1, 2, 3]))
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out


def _validate_conv_bn_relu_inputs(x, weight, bias, running_mean, running_var, gamma, beta) -> None:
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError('fused_conv_bn_relu expects x and weight in NCHW/OIHW 4D format')
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f'Input channels {x.shape[1]} do not match weight channels {weight.shape[1]}')
    out_channels = weight.shape[0]
    for name, arr in {
        'running_mean': running_mean,
        'running_var': running_var,
        'gamma': gamma,
        'beta': beta,
    }.items():
        if np.asarray(arr).shape != (out_channels,):
            raise ValueError(f'{name} must have shape ({out_channels},)')
    if bias is not None and np.asarray(bias).shape != (out_channels,):
        raise ValueError(f'bias must have shape ({out_channels},)')


def fused_conv_bn_relu(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    gamma,
    beta,
    *,
    padding: int = 0,
    eps: float = 1e-5,
    require_fusion: bool = False,
):
    _validate_conv_bn_relu_inputs(x, weight, bias, running_mean, running_var, gamma, beta)
    supported = can_use_fused_conv_bn_relu(kernel_size=weight.shape[-1], stride=1, groups=1, dtype=x.dtype)
    if not supported and require_fusion:
        raise ValueError('fused_conv_bn_relu supports only float32 NCHW, kernel_size=3, stride=1, groups=1')
    conv = _conv2d_nchw(x.astype(np.float32), weight.astype(np.float32), None if bias is None else bias.astype(np.float32), padding)
    scale, shift = compute_bn_affine(running_mean, running_var, gamma, beta, eps)
    y = conv * scale.reshape(1, -1, 1, 1) + shift.reshape(1, -1, 1, 1)
    return np.maximum(y, 0.0).astype(np.float32), {'fused': supported}
