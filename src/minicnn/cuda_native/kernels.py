"""Kernel registry and reference (numpy) kernels for cuda_native forward pass.

Phase 1 ships reference numpy kernels so the executor can run end-to-end
without a CUDA device.  Each kernel receives (node, context) where
context maps tensor names to numpy arrays and returns nothing — it
writes outputs into context in-place.
"""
from __future__ import annotations

from typing import Callable, Any

import numpy as np

from minicnn.cuda_native.nodes import Node


KernelFn = Callable[['Node', dict[str, Any]], None]


class KernelRegistry:
    """Maps op names to kernel callables."""

    def __init__(self) -> None:
        self._dispatch: dict[str, KernelFn] = {}

    def register(self, op_name: str, fn: KernelFn) -> 'KernelRegistry':
        self._dispatch[op_name] = fn
        return self

    def get(self, op_name: str) -> KernelFn:
        if op_name not in self._dispatch:
            raise KeyError(
                f'No cuda_native kernel registered for op: {op_name}. '
                f'Registered ops: {sorted(self._dispatch)}'
            )
        return self._dispatch[op_name]

    def has(self, op_name: str) -> bool:
        return op_name in self._dispatch

    def registered_ops(self) -> list[str]:
        return sorted(self._dispatch)


# ---------------------------------------------------------------------------
# Reference numpy kernels
# ---------------------------------------------------------------------------

def _kernel_conv2d(node: Node, ctx: dict[str, Any]) -> None:
    x: np.ndarray = ctx[node.inputs[0]]
    w: np.ndarray = ctx[f'_w_{node.name}']
    b: np.ndarray = ctx.get(f'_b_{node.name}')
    attrs = node.attrs
    stride = attrs.get('stride', 1)
    padding = attrs.get('padding', 0)
    sh = sw = stride if isinstance(stride, int) else stride[0]
    ph = pw = padding if isinstance(padding, int) else padding[0]

    n, c_in, h_in, w_in = x.shape
    c_out, _, kh, kw = w.shape
    h_out = (h_in + 2 * ph - kh) // sh + 1
    w_out = (w_in + 2 * pw - kw) // sw + 1

    if ph > 0 or pw > 0:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))

    out = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            patch = x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    if b is not None:
        out += b[None, :, None, None]
    ctx[node.outputs[0]] = out


def _kernel_batchnorm2d_eval(node: Node, ctx: dict[str, Any]) -> None:
    x: np.ndarray = ctx[node.inputs[0]]
    if x.ndim != 4:
        raise ValueError(
            f'BatchNorm2d expects 4-D input (N,C,H,W), got shape {x.shape}'
        )
    channels = x.shape[1]
    eps = float(node.attrs.get('eps', 1e-5))
    gamma = np.asarray(
        ctx.get(f'_w_{node.name}', np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    beta = np.asarray(
        ctx.get(f'_b_{node.name}', np.zeros(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    rm_key = f'_running_mean_{node.name}'
    rv_key = f'_running_var_{node.name}'
    running_mean = np.asarray(
        ctx.get(rm_key, np.zeros(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    running_var = np.asarray(
        ctx.get(rv_key, np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )

    for label, arr in (
        ('weight', gamma),
        ('bias', beta),
        ('running_mean', running_mean),
        ('running_var', running_var),
    ):
        if arr.shape != (channels,):
            raise ValueError(
                f'BatchNorm2d node={node.name}: {label} must have shape {(channels,)}, '
                f'got {arr.shape}'
            )

    mode = str(ctx.get('__mode__', 'eval'))
    if mode not in {'eval', 'train'}:
        raise ValueError(
            f'BatchNorm2d node={node.name}: unsupported execution mode {mode!r}; '
            "expected 'eval' or 'train'"
        )

    if mode == 'train':
        momentum = float(node.attrs.get('momentum', 0.1))
        batch_mean = x.mean(axis=(0, 2, 3)).astype(np.float32)
        batch_var = x.var(axis=(0, 2, 3)).astype(np.float32)
        next_mean = ((1.0 - momentum) * running_mean + momentum * batch_mean).astype(np.float32)
        next_var = ((1.0 - momentum) * running_var + momentum * batch_var).astype(np.float32)
        if rm_key in ctx:
            ctx[rm_key][...] = next_mean
        else:
            ctx[rm_key] = next_mean
        if rv_key in ctx:
            ctx[rv_key][...] = next_var
        else:
            ctx[rv_key] = next_var
        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var

    centered = x - mean[None, :, None, None]
    inv_std = 1.0 / np.sqrt(var[None, :, None, None] + eps)
    out = centered * inv_std
    out = out * gamma[None, :, None, None] + beta[None, :, None, None]
    ctx[node.outputs[0]] = out.astype(np.float32)


def _kernel_relu(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    ctx[node.outputs[0]] = np.maximum(x, 0.0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _kernel_leaky_relu(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    alpha = float(node.attrs.get('negative_slope', 0.01))
    ctx[node.outputs[0]] = np.where(x >= 0, x, alpha * x).astype(np.float32)


def _kernel_sigmoid(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    ctx[node.outputs[0]] = _sigmoid(x)


def _kernel_tanh(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    ctx[node.outputs[0]] = np.tanh(x).astype(np.float32)


def _kernel_silu(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    sig = _sigmoid(x)
    ctx[node.outputs[0]] = (x * sig).astype(np.float32)


def _kernel_flatten(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    ctx[node.outputs[0]] = x.reshape(x.shape[0], -1).astype(np.float32)


def _kernel_linear(node: Node, ctx: dict[str, Any]) -> None:
    x = ctx[node.inputs[0]]
    w: np.ndarray = ctx[f'_w_{node.name}']
    b: np.ndarray = ctx.get(f'_b_{node.name}')
    out = x @ w.T
    if b is not None:
        out = out + b
    ctx[node.outputs[0]] = out.astype(np.float32)


def _kernel_maxpool2d(node: Node, ctx: dict[str, Any]) -> None:
    x: np.ndarray = ctx[node.inputs[0]]
    attrs = node.attrs
    ks = attrs.get('kernel_size', 2)
    kh = kw = ks if isinstance(ks, int) else ks[0]
    st = attrs.get('stride', ks)
    sh = sw = st if isinstance(st, int) else st[0]
    n, c, h, w = x.shape
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    out = np.empty((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw].max(axis=(2, 3))
    ctx[node.outputs[0]] = out


def _kernel_avgpool2d(node: Node, ctx: dict[str, Any]) -> None:
    x: np.ndarray = ctx[node.inputs[0]]
    attrs = node.attrs
    ks = attrs.get('kernel_size', 2)
    kh = kw = ks if isinstance(ks, int) else ks[0]
    st = attrs.get('stride', ks)
    sh = sw = st if isinstance(st, int) else st[0]
    n, c, h, w = x.shape
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    out = np.empty((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw].mean(axis=(2, 3))
    ctx[node.outputs[0]] = out


def make_default_registry() -> KernelRegistry:
    """Build a KernelRegistry with all Phase-1/2 reference kernels."""
    reg = KernelRegistry()
    reg.register('Conv2d', _kernel_conv2d)
    reg.register('BatchNorm2d', _kernel_batchnorm2d_eval)
    reg.register('ReLU', _kernel_relu)
    reg.register('LeakyReLU', _kernel_leaky_relu)
    reg.register('Sigmoid', _kernel_sigmoid)
    reg.register('Tanh', _kernel_tanh)
    reg.register('SiLU', _kernel_silu)
    reg.register('Flatten', _kernel_flatten)
    reg.register('Linear', _kernel_linear)
    reg.register('MaxPool2d', _kernel_maxpool2d)
    reg.register('AvgPool2d', _kernel_avgpool2d)
    return reg
