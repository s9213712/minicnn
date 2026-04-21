"""Backward kernels and executor for cuda_native.

Each backward kernel receives:
    node       — the forward Node
    grad_out   — upstream gradient (same shape as node's output tensor)
    cache      — dict built by ForwardExecutor.run_with_cache()
    param_grads — dict to accumulate dL/dW, dL/db  (written in-place)

Returns grad_in: dL/d(node input), same shape as the node's input tensor.

Supported backward ops (Phase 3):
    Linear, ReLU, LeakyReLU, Sigmoid, Tanh, SiLU, Flatten, Conv2d,
    BatchNorm2d, MaxPool2d, AvgPool2d
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.nodes import Node


BwdKernelFn = Callable[[Node, np.ndarray, dict[str, Any], dict[str, Any]], np.ndarray]


# ---------------------------------------------------------------------------
# Backward kernel implementations
# ---------------------------------------------------------------------------

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
    w = cache[f'_w_{node.name}']        # (C_out, C_in, kH, kW)
    attrs = node.attrs
    stride = attrs.get('stride', 1)
    padding = attrs.get('padding', 0)
    sh = sw = stride if isinstance(stride, int) else stride[0]
    ph = pw = padding if isinstance(padding, int) else padding[0]

    n, c_in, h_in, w_in = x.shape
    c_out, _, kh, kw = w.shape
    _, _, h_out, w_out = grad_out.shape

    # dL/db
    b_key = f'_b_{node.name}'
    if b_key in cache:
        param_grads[b_key] = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)

    # Padded input for dL/dW
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x

    # dL/dW: (C_out, C_in, kH, kW)
    grad_w = np.zeros_like(w)
    for i in range(kh):
        for j in range(kw):
            # x_patch: (N, C_in, H_out, W_out) with stride
            x_patch = x_pad[:, :, i:i + h_out * sh:sh, j:j + w_out * sw:sw]
            # grad_w[:,  :, i, j] = einsum('nchw,ndhw->dc', grad_out, x_patch)
            # c=C_out d=C_in
            grad_w[:, :, i, j] = np.einsum('nchw,ndhw->cd', grad_out, x_patch)
    param_grads[f'_w_{node.name}'] = grad_w.astype(np.float32)

    # dL/dx: full convolution of grad_out with flipped w
    grad_x_pad = np.zeros((n, c_in, h_in + 2 * ph, w_in + 2 * pw), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            g = grad_out[:, :, i, j]           # (N, C_out)
            contrib = np.tensordot(g, w, axes=([1], [0]))  # (N, C_in, kH, kW)
            grad_x_pad[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw] += contrib

    if ph or pw:
        return grad_x_pad[:, :, ph:ph + h_in, pw:pw + w_in].astype(np.float32)
    return grad_x_pad.astype(np.float32)


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
    ).reshape(1, channels, 1, 1)

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

    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)

    if gamma_key in cache:
        param_grads[gamma_key] = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    if beta_key in cache:
        param_grads[beta_key] = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)

    if mode == 'eval':
        return (grad_out * gamma * inv_std).astype(np.float32)

    norm_elems = float(x.shape[0] * x.shape[2] * x.shape[3])
    sum_grad = grad_out.sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    sum_grad_xhat = (grad_out * x_hat).sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    grad_in = (
        (gamma * inv_std / norm_elems)
        * (norm_elems * grad_out - sum_grad - x_hat * sum_grad_xhat)
    )
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
    reg.register('ReLU', _bwd_relu)
    reg.register('LeakyReLU', _bwd_leaky_relu)
    reg.register('Sigmoid', _bwd_sigmoid)
    reg.register('Tanh', _bwd_tanh)
    reg.register('SiLU', _bwd_silu)
    reg.register('Flatten', _bwd_flatten)
    reg.register('Linear', _bwd_linear)
    reg.register('Conv2d', _bwd_conv2d)
    reg.register('BatchNorm2d', _bwd_batchnorm2d)
    reg.register('MaxPool2d', _bwd_maxpool2d)
    reg.register('AvgPool2d', _bwd_avgpool2d)
    return reg


# ---------------------------------------------------------------------------
# Backward executor
# ---------------------------------------------------------------------------

class BackwardExecutor:
    """Runs backward pass through a NativeGraph in reverse topological order."""

    def __init__(self, registry: BackwardRegistry | None = None) -> None:
        self.registry = registry if registry is not None else make_default_backward_registry()

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
        grad = grad_output

        for node in reversed(graph.topological_order()):
            bwd = self.registry.get(node.op_type)
            grad = bwd(node, grad, cache, param_grads)

        return grad, param_grads
