"""Backward kernels and executor for cuda_native.

Each backward kernel receives:
    node       — the forward Node
    grad_out   — upstream gradient (same shape as node's output tensor)
    cache      — dict built by ForwardExecutor.run_with_cache()
    param_grads — dict to accumulate dL/dW, dL/db  (written in-place)

Returns grad_in: dL/d(node input), same shape as the node's input tensor.

Supported backward ops (Phase 3):
    Linear, ReLU, LeakyReLU, Sigmoid, Tanh, SiLU, GELU, Identity,
    Flatten, Conv2d, DepthwiseConv2d, PointwiseConv2d, BatchNorm2d,
    GroupNorm, LayerNorm, LayerNorm2d, MaxPool2d, AvgPool2d,
    AdaptiveAvgPool2d, GlobalAvgPool2d
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.nodes import Node


BwdKernelFn = Callable[[Node, np.ndarray, dict[str, Any], dict[str, Any]], Any]


# ---------------------------------------------------------------------------
# Backward kernel implementations
# ---------------------------------------------------------------------------


def _conv2d_backward_arrays(
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sh = sw = stride if isinstance(stride, int) else stride[0]
    ph = pw = padding if isinstance(padding, int) else padding[0]
    n, c_in, h_in, w_in = x.shape
    c_out, w_in_per_group, kh, kw = w.shape
    _, _, h_out, w_out = grad_out.shape
    if c_in % groups != 0 or c_out % groups != 0:
        raise ValueError(
            f'Conv2d backward helper: invalid grouped shape with input channels={c_in}, '
            f'output channels={c_out}, groups={groups}'
        )
    in_per_group = c_in // groups
    out_per_group = c_out // groups
    if w_in_per_group != in_per_group:
        raise ValueError(
            f'[E_CONV2D_CHANNEL_GROUP_MISMATCH] Conv2d backward helper: '
            f'weight expects {w_in_per_group} input channels per group, '
            f'expected {in_per_group}'
        )
    grad_b = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x
    grad_w = np.zeros_like(w)
    for group_idx in range(groups):
        in_start = group_idx * in_per_group
        in_end = (group_idx + 1) * in_per_group
        out_start = group_idx * out_per_group
        out_end = (group_idx + 1) * out_per_group
        group_grad_out = grad_out[:, out_start:out_end, :, :]
        for i in range(kh):
            for j in range(kw):
                x_patch = x_pad[:, in_start:in_end, i:i + h_out * sh:sh, j:j + w_out * sw:sw]
                grad_w[out_start:out_end, :, i, j] = np.einsum(
                    'nchw,ndhw->cd',
                    group_grad_out,
                    x_patch,
                )
    grad_x_pad = np.zeros((n, c_in, h_in + 2 * ph, w_in + 2 * pw), dtype=np.float32)
    for group_idx in range(groups):
        in_start = group_idx * in_per_group
        in_end = (group_idx + 1) * in_per_group
        out_start = group_idx * out_per_group
        out_end = (group_idx + 1) * out_per_group
        group_weights = w[out_start:out_end, :, :, :]
        for i in range(h_out):
            for j in range(w_out):
                g = grad_out[:, out_start:out_end, i, j]
                contrib = np.tensordot(g, group_weights, axes=([1], [0]))
                grad_x_pad[:, in_start:in_end, i * sh:i * sh + kh, j * sw:j * sw + kw] += contrib
    if ph or pw:
        grad_x = grad_x_pad[:, :, ph:ph + h_in, pw:pw + w_in].astype(np.float32)
    else:
        grad_x = grad_x_pad.astype(np.float32)
    return grad_x, grad_w.astype(np.float32), grad_b.astype(np.float32)


def _batchnorm2d_backward_arrays(
    x: np.ndarray,
    grad_out: np.ndarray,
    *,
    gamma: np.ndarray,
    eps: float,
    mode: str,
    running_mean: np.ndarray,
    running_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = x.shape[1]
    gamma_r = np.asarray(gamma, dtype=np.float32).reshape(1, channels, 1, 1)
    if mode == 'train':
        mean = x.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        var = x.var(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    else:
        mean = np.asarray(running_mean, dtype=np.float32).reshape(1, channels, 1, 1)
        var = np.asarray(running_var, dtype=np.float32).reshape(1, channels, 1, 1)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    grad_gamma = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_beta = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)
    if mode == 'eval':
        return (grad_out * gamma_r * inv_std).astype(np.float32), grad_gamma, grad_beta
    norm_elems = float(x.shape[0] * x.shape[2] * x.shape[3])
    sum_grad = grad_out.sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    sum_grad_xhat = (grad_out * x_hat).sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    grad_in = (
        (gamma_r * inv_std / norm_elems)
        * (norm_elems * grad_out - sum_grad - x_hat * sum_grad_xhat)
    )
    return grad_in.astype(np.float32), grad_gamma, grad_beta


def _layernorm2d_backward_arrays(
    x: np.ndarray,
    grad_out: np.ndarray,
    *,
    gamma: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = x.shape[1]
    gamma_r = np.asarray(gamma, dtype=np.float32).reshape(1, channels, 1, 1)
    mean = x.mean(axis=1, keepdims=True).astype(np.float32)
    var = x.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    grad_gamma = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_beta = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = (grad_out * gamma_r).astype(np.float32)
    norm_elems = float(channels)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True).astype(np.float32)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True).astype(np.float32)
    grad_in = (
        (inv_std / norm_elems)
        * (norm_elems * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    )
    return grad_in.astype(np.float32), grad_gamma, grad_beta


def _groupnorm_backward_arrays(
    x: np.ndarray,
    grad_out: np.ndarray,
    *,
    gamma: np.ndarray,
    num_groups: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, c, h, w = x.shape
    if c % num_groups != 0:
        raise ValueError(
            f'GroupNorm backward expects num_groups={num_groups} to divide channels={c}'
        )
    channels_per_group = c // num_groups
    gamma_r = np.asarray(gamma, dtype=np.float32).reshape(1, c, 1, 1)
    x_group = x.reshape(n, num_groups, channels_per_group, h, w).astype(np.float32)
    grad_out_group = grad_out.reshape(n, num_groups, channels_per_group, h, w).astype(np.float32)
    mean = x_group.mean(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    var = x_group.var(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat_group = ((x_group - mean) * inv_std).astype(np.float32)
    x_hat = x_hat_group.reshape(n, c, h, w).astype(np.float32)
    grad_gamma = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_beta = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = (grad_out * gamma_r).reshape(n, num_groups, channels_per_group, h, w).astype(np.float32)
    norm_elems = float(channels_per_group * h * w)
    sum_dxhat = dxhat.sum(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    sum_dxhat_xhat = (dxhat * x_hat_group).sum(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    grad_group = (
        (inv_std / norm_elems)
        * (norm_elems * dxhat - sum_dxhat - x_hat_group * sum_dxhat_xhat)
    )
    return grad_group.reshape(n, c, h, w).astype(np.float32), grad_gamma, grad_beta


def _layernorm_backward_arrays(
    x: np.ndarray,
    grad_out: np.ndarray,
    *,
    gamma: np.ndarray,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not normalized_shape:
        raise ValueError('LayerNorm backward requires non-empty normalized_shape')
    if x.ndim < len(normalized_shape):
        raise ValueError(
            f'LayerNorm backward expects input rank >= {len(normalized_shape)}, got shape {x.shape}'
        )
    if tuple(int(v) for v in x.shape[-len(normalized_shape):]) != tuple(normalized_shape):
        raise ValueError(
            f'LayerNorm backward expected trailing shape {tuple(normalized_shape)}, got {tuple(x.shape[-len(normalized_shape):])}'
        )
    reduce_axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    keep_leading = tuple(range(x.ndim - len(normalized_shape)))
    mean = x.mean(axis=reduce_axes, keepdims=True).astype(np.float32)
    var = x.var(axis=reduce_axes, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    reshape = (1,) * (x.ndim - len(normalized_shape)) + tuple(normalized_shape)
    gamma_r = np.asarray(gamma, dtype=np.float32).reshape(reshape)
    grad_gamma = (grad_out * x_hat).sum(axis=keep_leading).astype(np.float32)
    grad_beta = grad_out.sum(axis=keep_leading).astype(np.float32)
    dxhat = (grad_out * gamma_r).astype(np.float32)
    norm_elems = float(np.prod(normalized_shape))
    sum_dxhat = dxhat.sum(axis=reduce_axes, keepdims=True).astype(np.float32)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=reduce_axes, keepdims=True).astype(np.float32)
    grad_in = (
        (inv_std / norm_elems)
        * (norm_elems * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    )
    return grad_in.astype(np.float32), grad_gamma, grad_beta


def _bwd_relu(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    return (grad_out * (x > 0)).astype(np.float32)


def _bwd_leaky_relu(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    alpha = float(node.attrs.get('negative_slope', 0.01))
    mask = np.where(x >= 0, 1.0, alpha).astype(np.float32)
    return (grad_out * mask).astype(np.float32)


def _bwd_sigmoid(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    sig = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    return (grad_out * sig * (1.0 - sig)).astype(np.float32)


def _bwd_tanh(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    y = np.tanh(x).astype(np.float32)
    return (grad_out * (1.0 - y * y)).astype(np.float32)


def _bwd_silu(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    sig = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    return (grad_out * (sig + x * sig * (1.0 - sig))).astype(np.float32)


def _bwd_gelu(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    tanh_inner = np.tanh(inner).astype(np.float32)
    sech2_inner = (1.0 - tanh_inner * tanh_inner).astype(np.float32)
    inner_grad = (
        np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
    ).astype(np.float32)
    grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * inner_grad
    return (grad_out * grad).astype(np.float32)


def _bwd_identity(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    return grad_out.astype(np.float32)


def _bwd_add(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> list[np.ndarray]:
    return [np.asarray(grad_out, dtype=np.float32).copy() for _ in node.inputs]


def _bwd_concat(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> list[np.ndarray]:
    input_tensors = cache.get(f'fwd_{node.name}_inputs', [])
    if len(input_tensors) != len(node.inputs):
        raise ValueError(
            f'Concat backward node={node.name}: expected cached inputs for all tensors, got {len(input_tensors)}'
        )
    axis = int(node.attrs.get('axis', 1))
    split_sizes = [np.asarray(arr, dtype=np.float32).shape[axis] for arr in input_tensors]
    split_points = np.cumsum(split_sizes[:-1], dtype=np.int64)
    if len(split_points) == 0:
        return [np.asarray(grad_out, dtype=np.float32)]
    return [part.astype(np.float32) for part in np.split(grad_out, split_points, axis=axis)]


def _bwd_groupnorm(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    if x.ndim != 4:
        raise ValueError(
            f'GroupNorm backward expects 4-D cached input (N,C,H,W), got shape {x.shape}'
        )
    channels = x.shape[1]
    num_groups = int(node.attrs.get('num_groups', 0))
    if num_groups <= 0:
        raise ValueError(
            f'GroupNorm backward node={node.name}: attr "num_groups" must be > 0, got {num_groups}'
        )
    eps = float(node.attrs.get('eps', 1e-5))
    gamma_key = f'_w_{node.name}'
    beta_key = f'_b_{node.name}'
    gamma = np.asarray(
        cache.get(gamma_key, np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    grad_in, grad_gamma, grad_beta = _groupnorm_backward_arrays(
        x,
        grad_out,
        gamma=gamma,
        num_groups=num_groups,
        eps=eps,
    )
    if gamma_key in cache:
        param_grads[gamma_key] = grad_gamma.astype(np.float32)
    if beta_key in cache:
        param_grads[beta_key] = grad_beta.astype(np.float32)
    return grad_in.astype(np.float32)


def _bwd_layernorm(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = np.asarray(cache[f'fwd_{node.name}_in'], dtype=np.float32)
    raw_shape = node.attrs.get('normalized_shape')
    if raw_shape is None:
        raise ValueError(
            f'LayerNorm backward node={node.name}: missing required attr "normalized_shape"'
        )
    normalized_shape = (
        (int(raw_shape),)
        if isinstance(raw_shape, int)
        else tuple(int(v) for v in raw_shape)
    )
    eps = float(node.attrs.get('eps', 1e-5))
    gamma_key = f'_w_{node.name}'
    beta_key = f'_b_{node.name}'
    gamma = np.asarray(
        cache.get(gamma_key, np.ones(normalized_shape, dtype=np.float32)),
        dtype=np.float32,
    )
    grad_in, grad_gamma, grad_beta = _layernorm_backward_arrays(
        x,
        grad_out,
        gamma=gamma,
        normalized_shape=normalized_shape,
        eps=eps,
    )
    if gamma_key in cache:
        param_grads[gamma_key] = grad_gamma.astype(np.float32)
    if beta_key in cache:
        param_grads[beta_key] = grad_beta.astype(np.float32)
    return grad_in.astype(np.float32)


def _bwd_dropout(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    node_cache = cache.get(f'__cache_{node.name}', {})
    mask = np.asarray(node_cache.get('mask', np.ones_like(grad_out, dtype=np.float32)), dtype=np.float32)
    return (grad_out * mask).astype(np.float32)


def _bwd_droppath(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    node_cache = cache.get(f'__cache_{node.name}', {})
    mask = np.asarray(node_cache.get('mask', np.ones_like(grad_out, dtype=np.float32)), dtype=np.float32)
    return (grad_out * mask).astype(np.float32)


def _bwd_flatten(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    in_shape = cache[f'fwd_{node.name}_in_shape']
    return grad_out.reshape(in_shape).astype(np.float32)


def _bwd_linear(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']   # (N, in_features)
    w = cache[f'_w_{node.name}']        # (out_features, in_features)
    # dL/dW = grad_out.T @ x  ->  (out_features, in_features)
    param_grads[f'_w_{node.name}'] = (grad_out.T @ x).astype(np.float32)
    b_key = f'_b_{node.name}'
    if b_key in cache:
        param_grads[b_key] = grad_out.sum(axis=0).astype(np.float32)
    # dL/dx = grad_out @ w  ->  (N, in_features)
    return (grad_out @ w).astype(np.float32)


def _bwd_conv2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']   # (N, C_in, H_in, W_in)
    w = cache[f'_w_{node.name}']        # (C_out, C_in/groups, kH, kW)
    attrs = node.attrs
    stride = attrs.get('stride', 1)
    padding = attrs.get('padding', 0)
    sh = sw = stride if isinstance(stride, int) else stride[0]
    ph = pw = padding if isinstance(padding, int) else padding[0]
    groups = int(attrs.get('groups', x.shape[1] if node.op_type in {'DepthwiseConv2d', 'depthwise_conv2d'} else 1))

    grad_x, grad_w, grad_b = _conv2d_backward_arrays(
        x,
        w,
        grad_out,
        stride=(sh, sw),
        padding=(ph, pw),
        groups=groups,
    )
    param_grads[f'_w_{node.name}'] = grad_w.astype(np.float32)
    b_key = f'_b_{node.name}'
    if b_key in cache:
        param_grads[b_key] = grad_b.astype(np.float32)
    return grad_x.astype(np.float32)


def _bwd_batchnorm2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    if x.ndim != 4:
        raise ValueError(
            f'BatchNorm2d backward expects 4-D cached input (N,C,H,W), got shape {x.shape}'
        )

    channels = x.shape[1]
    eps = float(node.attrs.get('eps', 1e-5))
    mode = str(cache.get(f'fwd_{node.name}_mode', 'eval'))
    gamma_key = f'_w_{node.name}'
    beta_key = f'_b_{node.name}'
    running_mean_key = f'_running_mean_{node.name}'
    running_var_key = f'_running_var_{node.name}'

    gamma = np.asarray(
        cache.get(gamma_key, np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )

    if mode == 'train':
        mean = x.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        var = x.var(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    elif mode == 'eval':
        mean = np.asarray(
            cache.get(running_mean_key, np.zeros(channels, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1, channels, 1, 1)
        var = np.asarray(
            cache.get(running_var_key, np.ones(channels, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1, channels, 1, 1)
    else:
        raise ValueError(
            f'BatchNorm2d backward does not support cached mode {mode!r}; '
            "expected 'eval' or 'train'"
        )

    grad_in, grad_gamma, grad_beta = _batchnorm2d_backward_arrays(
        x,
        grad_out,
        gamma=gamma,
        eps=eps,
        mode=mode,
        running_mean=np.asarray(cache.get(running_mean_key, np.zeros(channels, dtype=np.float32)), dtype=np.float32),
        running_var=np.asarray(cache.get(running_var_key, np.ones(channels, dtype=np.float32)), dtype=np.float32),
    )
    if gamma_key in cache:
        param_grads[gamma_key] = grad_gamma.astype(np.float32)
    if beta_key in cache:
        param_grads[beta_key] = grad_beta.astype(np.float32)
    return grad_in.astype(np.float32)


def _bwd_layernorm2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    if x.ndim != 4:
        raise ValueError(
            f'LayerNorm2d backward expects 4-D cached input (N,C,H,W), got shape {x.shape}'
        )
    channels = x.shape[1]
    eps = float(node.attrs.get('eps', 1e-6))
    gamma_key = f'_w_{node.name}'
    beta_key = f'_b_{node.name}'
    gamma = np.asarray(
        cache.get(gamma_key, np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    grad_in, grad_gamma, grad_beta = _layernorm2d_backward_arrays(
        x,
        grad_out,
        gamma=gamma,
        eps=eps,
    )
    if gamma_key in cache:
        param_grads[gamma_key] = grad_gamma.astype(np.float32)
    if beta_key in cache:
        param_grads[beta_key] = grad_beta.astype(np.float32)
    return grad_in.astype(np.float32)


def _bwd_maxpool2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']   # (N, C, H_in, W_in)
    attrs = node.attrs
    ks = attrs.get('kernel_size', 2)
    kh = kw = ks if isinstance(ks, int) else ks[0]
    st = attrs.get('stride', ks)
    sh = sw = st if isinstance(st, int) else st[0]
    n, c, h_in, w_in = x.shape
    h_out, w_out = grad_out.shape[2], grad_out.shape[3]
    grad_x = np.zeros_like(x)
    for i in range(h_out):
        for j in range(w_out):
            patch = x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]   # (N,C,kH,kW)
            max_val = patch.max(axis=(2, 3), keepdims=True)              # (N,C,1,1)
            mask = (patch == max_val).astype(np.float32)
            # distribute grad equally among tied maxima
            mask /= mask.sum(axis=(2, 3), keepdims=True).clip(min=1)
            grad_x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw] += (
                mask * grad_out[:, :, i:i + 1, j:j + 1]
            )
    return grad_x.astype(np.float32)


def _bwd_avgpool2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    attrs = node.attrs
    ks = attrs.get('kernel_size', 2)
    kh = kw = ks if isinstance(ks, int) else ks[0]
    st = attrs.get('stride', ks)
    sh = sw = st if isinstance(st, int) else st[0]
    h_out, w_out = grad_out.shape[2], grad_out.shape[3]
    grad_x = np.zeros_like(x)
    pool_area = kh * kw
    for i in range(h_out):
        for j in range(w_out):
            grad_x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw] += (
                grad_out[:, :, i:i + 1, j:j + 1] / pool_area
            )
    return grad_x.astype(np.float32)


def _bwd_global_avgpool2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    x = cache[f'fwd_{node.name}_in']
    if x.ndim != 4:
        raise ValueError(
            f'{node.op_type} backward expects 4-D cached input (N,C,H,W), got shape {x.shape}'
        )
    spatial_area = float(x.shape[2] * x.shape[3])
    return np.broadcast_to(grad_out / spatial_area, x.shape).astype(np.float32)


def _bwd_adaptive_avgpool2d(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    output_size = node.attrs.get('output_size', 1)
    normalized = tuple(output_size) if isinstance(output_size, (list, tuple)) else output_size
    if normalized not in {1, (1, 1)}:
        raise ValueError(
            f'AdaptiveAvgPool2d backward node={node.name}: only output_size=1 or (1, 1) is supported, got {output_size!r}'
    )
    return _bwd_global_avgpool2d(node, grad_out, cache, param_grads)


def _bwd_residual_block(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    node_cache = cache[f'__cache_{node.name}']
    x = np.asarray(node_cache['x'], dtype=np.float32)
    conv1 = np.asarray(node_cache['conv1'], dtype=np.float32)
    bn1 = np.asarray(node_cache['bn1'], dtype=np.float32)
    act1 = np.asarray(node_cache['act1'], dtype=np.float32)
    shortcut = np.asarray(node_cache['shortcut'], dtype=np.float32)
    summed = np.asarray(node_cache['summed'], dtype=np.float32)
    use_shortcut_conv = bool(int(np.asarray(node_cache['use_shortcut_conv']).item()))
    stride = int(node.attrs.get('stride', 1))
    kernel_size = int(node.attrs.get('kernel_size', 3))
    padding = int(node.attrs.get('padding', kernel_size // 2))
    bias = bool(node.attrs.get('bias', False))
    eps = float(node.attrs.get('eps', 1e-5))
    mode = 'train' if bool(int(np.asarray(node_cache['bn1_cache']['mode']).item())) else 'eval'

    grad_summed = (grad_out * (summed > 0)).astype(np.float32)
    grad_bn2 = grad_summed
    grad_shortcut = grad_summed

    gamma_bn2 = np.asarray(cache[f'_w_bn2_{node.name}'], dtype=np.float32)
    grad_conv2, grad_gamma_bn2, grad_beta_bn2 = _batchnorm2d_backward_arrays(
        np.asarray(node_cache['conv2'], dtype=np.float32),
        grad_bn2,
        gamma=gamma_bn2,
        eps=eps,
        mode=mode,
        running_mean=np.asarray(cache[f'_running_mean_bn2_{node.name}'], dtype=np.float32),
        running_var=np.asarray(cache[f'_running_var_bn2_{node.name}'], dtype=np.float32),
    )
    param_grads[f'_w_bn2_{node.name}'] = grad_gamma_bn2
    param_grads[f'_b_bn2_{node.name}'] = grad_beta_bn2

    grad_act1, grad_w_conv2, grad_b_conv2 = _conv2d_backward_arrays(
        act1,
        np.asarray(cache[f'_w_conv2_{node.name}'], dtype=np.float32),
        grad_conv2,
        stride=1,
        padding=padding,
        groups=1,
    )
    param_grads[f'_w_conv2_{node.name}'] = grad_w_conv2
    if bias and f'_b_conv2_{node.name}' in cache:
        param_grads[f'_b_conv2_{node.name}'] = grad_b_conv2

    grad_bn1 = (grad_act1 * (bn1 > 0)).astype(np.float32)
    gamma_bn1 = np.asarray(cache[f'_w_bn1_{node.name}'], dtype=np.float32)
    grad_conv1, grad_gamma_bn1, grad_beta_bn1 = _batchnorm2d_backward_arrays(
        conv1,
        grad_bn1,
        gamma=gamma_bn1,
        eps=eps,
        mode=mode,
        running_mean=np.asarray(cache[f'_running_mean_bn1_{node.name}'], dtype=np.float32),
        running_var=np.asarray(cache[f'_running_var_bn1_{node.name}'], dtype=np.float32),
    )
    param_grads[f'_w_bn1_{node.name}'] = grad_gamma_bn1
    param_grads[f'_b_bn1_{node.name}'] = grad_beta_bn1

    grad_main, grad_w_conv1, grad_b_conv1 = _conv2d_backward_arrays(
        x,
        np.asarray(cache[f'_w_conv1_{node.name}'], dtype=np.float32),
        grad_conv1,
        stride=stride,
        padding=padding,
        groups=1,
    )
    param_grads[f'_w_conv1_{node.name}'] = grad_w_conv1
    if bias and f'_b_conv1_{node.name}' in cache:
        param_grads[f'_b_conv1_{node.name}'] = grad_b_conv1

    if use_shortcut_conv:
        gamma_short = np.asarray(cache[f'_w_shortcut_bn_{node.name}'], dtype=np.float32)
        grad_shortcut_conv, grad_gamma_short, grad_beta_short = _batchnorm2d_backward_arrays(
            np.asarray(node_cache['shortcut_conv'], dtype=np.float32),
            grad_shortcut,
            gamma=gamma_short,
            eps=eps,
            mode=mode,
            running_mean=np.asarray(cache[f'_running_mean_shortcut_bn_{node.name}'], dtype=np.float32),
            running_var=np.asarray(cache[f'_running_var_shortcut_bn_{node.name}'], dtype=np.float32),
        )
        param_grads[f'_w_shortcut_bn_{node.name}'] = grad_gamma_short
        param_grads[f'_b_shortcut_bn_{node.name}'] = grad_beta_short
        grad_short, grad_w_short, _grad_b_short = _conv2d_backward_arrays(
            x,
            np.asarray(cache[f'_w_shortcut_conv_{node.name}'], dtype=np.float32),
            grad_shortcut_conv,
            stride=stride,
            padding=0,
            groups=1,
        )
        param_grads[f'_w_shortcut_conv_{node.name}'] = grad_w_short
    else:
        grad_short = grad_shortcut.astype(np.float32)

    return (grad_main + grad_short).astype(np.float32)


def _bwd_convnext_block(node: Node, grad_out: np.ndarray, cache: dict, param_grads: dict) -> np.ndarray:
    node_cache = cache[f'__cache_{node.name}']
    x = np.asarray(node_cache['x'], dtype=np.float32)
    depthwise = np.asarray(node_cache['depthwise'], dtype=np.float32)
    norm = np.asarray(node_cache['norm'], dtype=np.float32)
    pw1 = np.asarray(node_cache['pw1'], dtype=np.float32)
    activated = np.asarray(node_cache['activated'], dtype=np.float32)
    has_layer_scale = bool(int(np.asarray(node_cache['has_layer_scale']).item()))
    bias = bool(node.attrs.get('bias', True))
    eps = float(node.attrs.get('layer_norm_eps', 1e-6))
    kernel_size = int(node.attrs.get('kernel_size', 7))
    padding = kernel_size // 2

    grad_x = grad_out.astype(np.float32)
    grad_scaled = grad_out.astype(np.float32)
    if has_layer_scale:
        layer_scale = np.asarray(cache[f'_layer_scale_{node.name}'], dtype=np.float32).reshape(1, -1, 1, 1)
        pw2 = np.asarray(node_cache['pw2'], dtype=np.float32)
        param_grads[f'_layer_scale_{node.name}'] = (grad_scaled * pw2).sum(axis=(0, 2, 3)).astype(np.float32)
        grad_pw2 = (grad_scaled * layer_scale).astype(np.float32)
    else:
        grad_pw2 = grad_scaled.astype(np.float32)

    grad_activated, grad_w_pw2, grad_b_pw2 = _conv2d_backward_arrays(
        activated,
        np.asarray(cache[f'_w_pw2_{node.name}'], dtype=np.float32),
        grad_pw2,
        stride=1,
        padding=0,
        groups=1,
    )
    param_grads[f'_w_pw2_{node.name}'] = grad_w_pw2
    if bias and f'_b_pw2_{node.name}' in cache:
        param_grads[f'_b_pw2_{node.name}'] = grad_b_pw2

    inner = np.sqrt(2.0 / np.pi) * (pw1 + 0.044715 * np.power(pw1, 3))
    tanh_inner = np.tanh(inner).astype(np.float32)
    sech2_inner = (1.0 - tanh_inner * tanh_inner).astype(np.float32)
    inner_grad = (
        np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(pw1, 2))
    ).astype(np.float32)
    gelu_grad = (0.5 * (1.0 + tanh_inner) + 0.5 * pw1 * sech2_inner * inner_grad).astype(np.float32)
    grad_pw1 = (grad_activated * gelu_grad).astype(np.float32)

    grad_norm, grad_w_pw1, grad_b_pw1 = _conv2d_backward_arrays(
        norm,
        np.asarray(cache[f'_w_pw1_{node.name}'], dtype=np.float32),
        grad_pw1,
        stride=1,
        padding=0,
        groups=1,
    )
    param_grads[f'_w_pw1_{node.name}'] = grad_w_pw1
    if bias and f'_b_pw1_{node.name}' in cache:
        param_grads[f'_b_pw1_{node.name}'] = grad_b_pw1

    grad_depthwise, grad_gamma_ln, grad_beta_ln = _layernorm2d_backward_arrays(
        depthwise,
        grad_norm,
        gamma=np.asarray(cache[f'_w_ln_{node.name}'], dtype=np.float32),
        eps=eps,
    )
    param_grads[f'_w_ln_{node.name}'] = grad_gamma_ln
    param_grads[f'_b_ln_{node.name}'] = grad_beta_ln

    grad_residual, grad_w_depthwise, grad_b_depthwise = _conv2d_backward_arrays(
        x,
        np.asarray(cache[f'_w_depthwise_{node.name}'], dtype=np.float32),
        grad_depthwise,
        stride=1,
        padding=padding,
        groups=x.shape[1],
    )
    param_grads[f'_w_depthwise_{node.name}'] = grad_w_depthwise
    if bias and f'_b_depthwise_{node.name}' in cache:
        param_grads[f'_b_depthwise_{node.name}'] = grad_b_depthwise

    return (grad_x + grad_residual).astype(np.float32)


# ---------------------------------------------------------------------------
# Backward kernel registry
# ---------------------------------------------------------------------------

class BackwardRegistry:
    def __init__(self) -> None:
        self._dispatch: dict[str, BwdKernelFn] = {}

    def register(self, op_name: str, fn: BwdKernelFn) -> 'BackwardRegistry':
        self._dispatch[op_name] = fn
        return self

    def get(self, op_name: str) -> BwdKernelFn:
        if op_name not in self._dispatch:
            raise KeyError(
                f'No cuda_native backward kernel for op: {op_name}. '
                f'Registered: {sorted(self._dispatch)}'
            )
        return self._dispatch[op_name]

    def has(self, op_name: str) -> bool:
        return op_name in self._dispatch


def make_default_backward_registry() -> BackwardRegistry:
    reg = BackwardRegistry()
    reg.register('Add', _bwd_add)
    reg.register('Concat', _bwd_concat)
    reg.register('ReLU', _bwd_relu)
    reg.register('LeakyReLU', _bwd_leaky_relu)
    reg.register('Sigmoid', _bwd_sigmoid)
    reg.register('Tanh', _bwd_tanh)
    reg.register('SiLU', _bwd_silu)
    reg.register('GELU', _bwd_gelu)
    reg.register('Identity', _bwd_identity)
    reg.register('Dropout', _bwd_dropout)
    reg.register('DropPath', _bwd_droppath)
    reg.register('Flatten', _bwd_flatten)
    reg.register('Linear', _bwd_linear)
    reg.register('Conv2d', _bwd_conv2d)
    reg.register('DepthwiseConv2d', _bwd_conv2d)
    reg.register('depthwise_conv2d', _bwd_conv2d)
    reg.register('PointwiseConv2d', _bwd_conv2d)
    reg.register('pointwise_conv2d', _bwd_conv2d)
    reg.register('ResidualBlock', _bwd_residual_block)
    reg.register('ConvNeXtBlock', _bwd_convnext_block)
    reg.register('convnext_block', _bwd_convnext_block)
    reg.register('BatchNorm2d', _bwd_batchnorm2d)
    reg.register('GroupNorm', _bwd_groupnorm)
    reg.register('LayerNorm', _bwd_layernorm)
    reg.register('LayerNorm2d', _bwd_layernorm2d)
    reg.register('layernorm2d', _bwd_layernorm2d)
    reg.register('MaxPool2d', _bwd_maxpool2d)
    reg.register('AvgPool2d', _bwd_avgpool2d)
    reg.register('AdaptiveAvgPool2d', _bwd_adaptive_avgpool2d)
    reg.register('GlobalAvgPool2d', _bwd_global_avgpool2d)
    return reg


# ---------------------------------------------------------------------------
# Backward executor
# ---------------------------------------------------------------------------

class BackwardExecutor:
    """Runs backward pass through a NativeGraph in reverse topological order."""

    def __init__(self, registry: BackwardRegistry | None = None) -> None:
        self.registry = registry if registry is not None else make_default_backward_registry()

    @staticmethod
    def _normalize_input_grads(node: Node, grad_result: Any) -> list[np.ndarray]:
        if isinstance(grad_result, np.ndarray):
            if len(node.inputs) != 1:
                raise ValueError(
                    f'Backward kernel for {node.op_type} node={node.name} returned a single '
                    f'gradient for {len(node.inputs)} inputs'
                )
            return [grad_result.astype(np.float32)]
        if isinstance(grad_result, (list, tuple)):
            if len(grad_result) != len(node.inputs):
                raise ValueError(
                    f'Backward kernel for {node.op_type} node={node.name} returned '
                    f'{len(grad_result)} gradients for {len(node.inputs)} inputs'
                )
            return [np.asarray(grad, dtype=np.float32) for grad in grad_result]
        raise TypeError(
            f'Backward kernel for {node.op_type} node={node.name} returned unsupported '
            f'gradient container {type(grad_result)!r}'
        )

    def run(
        self,
        graph: NativeGraph,
        grad_output: np.ndarray,
        cache: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute gradients for all parameters and the graph input.

        Args:
            graph:       NativeGraph (same one used in the forward pass).
            grad_output: dL/d(graph output), shape == graph.output_spec.shape.
            cache:       dict returned by ForwardExecutor.run_with_cache().

        Returns:
            (grad_input, param_grads) where:
              grad_input  — dL/d(graph input)
              param_grads — dict mapping '_w_{node}' / '_b_{node}' -> gradient array
        """
        param_grads: dict[str, np.ndarray] = {}
        if graph.output_spec is None:
            raise ValueError('Graph has no output_spec; was it built with build_graph()?')
        if graph.input_spec is None:
            raise ValueError('Graph has no input_spec; was it built with build_graph()?')
        tensor_grads: dict[str, np.ndarray] = {
            graph.output_spec.name: np.asarray(grad_output, dtype=np.float32)
        }

        for node in reversed(graph.topological_order()):
            grad_out: np.ndarray | None = None
            for output_name in node.outputs:
                node_grad = tensor_grads.pop(output_name, None)
                if node_grad is None:
                    continue
                grad_out = node_grad if grad_out is None else (grad_out + node_grad).astype(np.float32)
            if grad_out is None:
                continue
            bwd = self.registry.get(node.op_type)
            input_grads = self._normalize_input_grads(node, bwd(node, grad_out, cache, param_grads))
            for input_name, input_grad in zip(node.inputs, input_grads):
                if input_name in tensor_grads:
                    tensor_grads[input_name] = (tensor_grads[input_name] + input_grad).astype(np.float32)
                else:
                    tensor_grads[input_name] = input_grad.astype(np.float32)

        grad_input = tensor_grads.get(graph.input_spec.name)
        if grad_input is None:
            raise ValueError(
                f'No gradient reached graph input tensor {graph.input_spec.name!r}; '
                'the graph may be disconnected.'
            )
        return grad_input.astype(np.float32), param_grads
