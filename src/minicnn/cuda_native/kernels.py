"""Kernel registry and reference (numpy) kernels for cuda_native forward pass.

Phase 1 ships reference numpy kernels so the executor can run end-to-end
without a CUDA device.  Each kernel receives (node, context) where
context maps tensor names to numpy arrays and returns nothing — it
writes outputs into context in-place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from minicnn.cuda_native.nodes import Node


KernelFn = Callable[['Node', dict[str, Any]], None]


@dataclass(frozen=True)
class KernelSpec:
    """Describes one registered kernel and its dispatch metadata."""

    op_name: str
    fn: KernelFn
    category: str

    def __iter__(self) -> Iterator[Any]:
        """Preserve tuple-unpack compatibility for existing call sites."""
        yield self.op_name
        yield self.fn


class KernelRegistry:
    """Maps op names to kernel callables."""

    def __init__(self) -> None:
        self._dispatch: dict[str, KernelFn] = {}
        self._specs: dict[str, KernelSpec] = {}

    def register(
        self,
        op_name: str,
        fn: KernelFn,
        *,
        category: str = 'generic',
    ) -> 'KernelRegistry':
        spec = KernelSpec(op_name=op_name, fn=fn, category=category)
        self._dispatch[op_name] = fn
        self._specs[op_name] = spec
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

    def spec(self, op_name: str) -> KernelSpec:
        if op_name not in self._specs:
            raise KeyError(
                f'No cuda_native kernel registered for op: {op_name}. '
                f'Registered ops: {sorted(self._specs)}'
            )
        return self._specs[op_name]

    def describe(self, op_name: str) -> dict[str, str]:
        spec = self.spec(op_name)
        return {
            'op_name': spec.op_name,
            'category': spec.category,
        }

    def registered_specs(self) -> list[KernelSpec]:
        return [self._specs[op_name] for op_name in self.registered_ops()]


# ---------------------------------------------------------------------------
# Reference numpy kernels
# ---------------------------------------------------------------------------


def _attr_pair(value: int | tuple[int, int] | list[int], *, label: str, node: Node) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f'{node.op_type} node={node.name}: {label} must be an int or length-2 pair, '
                f'got {value!r}'
            )
        first, second = value
    else:
        first = second = value
    try:
        first_i = int(first)
        second_i = int(second)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{node.op_type} node={node.name}: {label} must contain integers, got {value!r}'
        ) from exc
    return first_i, second_i


def _pool2d_windows(node: Node, x: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    if x.ndim != 4:
        raise ValueError(
            f'{node.op_type} expects 4-D input (N,C,H,W), got shape {x.shape}'
        )
    attrs = node.attrs
    kh, kw = _attr_pair(attrs.get('kernel_size', 2), label='kernel_size', node=node)
    if kh <= 0 or kw <= 0:
        raise ValueError(
            f'{node.op_type} node={node.name}: kernel_size must be positive, got {(kh, kw)}'
        )
    stride_value = attrs.get('stride', (kh, kw))
    sh, sw = _attr_pair(stride_value, label='stride', node=node)
    if sh <= 0 or sw <= 0:
        raise ValueError(
            f'{node.op_type} node={node.name}: stride must be positive, got {(sh, sw)}'
        )
    ph, pw = _attr_pair(attrs.get('padding', 0), label='padding', node=node)
    if ph < 0 or pw < 0:
        raise ValueError(
            f'{node.op_type} node={node.name}: padding must be non-negative, got {(ph, pw)}'
        )
    if ph > 0 or pw > 0:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    windows = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw, :, :]
    if windows.shape[2] == 0 or windows.shape[3] == 0:
        raise ValueError(
            f'{node.op_type} node={node.name}: invalid output shape for '
            f'input={x.shape}, kernel={(kh, kw)}, stride={(sh, sw)}, padding={(ph, pw)}'
        )
    return windows, (kh, kw), (sh, sw)


def _kernel_pool2d_reduce(
    node: Node,
    ctx: dict[str, Any],
    *,
    reducer: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    windows, _kernel, _stride = _pool2d_windows(node, x)
    ctx[node.outputs[0]] = reducer(windows).astype(np.float32)


def _kernel_conv2d(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    w = np.asarray(ctx[f'_w_{node.name}'], dtype=np.float32)
    b = ctx.get(f'_b_{node.name}')
    if x.ndim != 4:
        raise ValueError(f'Conv2d expects 4-D input (N,C,H,W), got shape {x.shape}')
    if w.ndim != 4:
        raise ValueError(f'Conv2d node={node.name}: weight must be 4-D, got shape {w.shape}')
    sh, sw = _attr_pair(node.attrs.get('stride', 1), label='stride', node=node)
    ph, pw = _attr_pair(node.attrs.get('padding', 0), label='padding', node=node)
    if sh <= 0 or sw <= 0:
        raise ValueError(f'Conv2d node={node.name}: stride must be positive, got {(sh, sw)}')
    if ph < 0 or pw < 0:
        raise ValueError(f'Conv2d node={node.name}: padding must be non-negative, got {(ph, pw)}')

    _n, c_in, _h_in, _w_in = x.shape
    c_out, w_in, kh, kw = w.shape
    if w_in != c_in:
        raise ValueError(
            f'Conv2d node={node.name}: weight expects {w_in} input channels, got {c_in}'
        )

    if ph > 0 or pw > 0:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))

    windows = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw, :, :]
    if windows.shape[2] == 0 or windows.shape[3] == 0:
        raise ValueError(
            f'Conv2d node={node.name}: invalid output shape for input={x.shape}, '
            f'kernel={(kh, kw)}, stride={(sh, sw)}, padding={(ph, pw)}'
        )
    out = np.tensordot(
        windows,
        w,
        axes=([1, 4, 5], [1, 2, 3]),
    ).transpose(0, 3, 1, 2)
    if b is not None:
        b_arr = np.asarray(b, dtype=np.float32)
        if b_arr.shape != (c_out,):
            raise ValueError(
                f'Conv2d node={node.name}: bias must have shape {(c_out,)}, got {b_arr.shape}'
            )
        out = out + b_arr[None, :, None, None]
    ctx[node.outputs[0]] = out.astype(np.float32)


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _kernel_elementwise(
    node: Node,
    ctx: dict[str, Any],
    *,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    ctx[node.outputs[0]] = transform(x).astype(np.float32)


def _kernel_relu(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_elementwise(node, ctx, transform=lambda x: np.maximum(x, 0.0))


def _kernel_leaky_relu(node: Node, ctx: dict[str, Any]) -> None:
    alpha = float(node.attrs.get('negative_slope', 0.01))
    _kernel_elementwise(node, ctx, transform=lambda x: np.where(x >= 0, x, alpha * x))


def _kernel_sigmoid(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_elementwise(node, ctx, transform=_sigmoid)


def _kernel_tanh(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_elementwise(node, ctx, transform=np.tanh)


def _kernel_silu(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_elementwise(node, ctx, transform=lambda x: x * _sigmoid(x))


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
    _kernel_pool2d_reduce(
        node,
        ctx,
        reducer=lambda windows: windows.max(axis=(-2, -1)),
    )


def _kernel_avgpool2d(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_pool2d_reduce(
        node,
        ctx,
        reducer=lambda windows: windows.mean(axis=(-2, -1)),
    )


_ACTIVATION_KERNEL_SPECS: tuple[KernelSpec, ...] = (
    KernelSpec('ReLU', _kernel_relu, 'activation'),
    KernelSpec('LeakyReLU', _kernel_leaky_relu, 'activation'),
    KernelSpec('Sigmoid', _kernel_sigmoid, 'activation'),
    KernelSpec('Tanh', _kernel_tanh, 'activation'),
    KernelSpec('SiLU', _kernel_silu, 'activation'),
)


DEFAULT_KERNEL_SPECS: tuple[KernelSpec, ...] = (
    KernelSpec('Conv2d', _kernel_conv2d, 'convolution'),
    KernelSpec('BatchNorm2d', _kernel_batchnorm2d_eval, 'normalization'),
    *_ACTIVATION_KERNEL_SPECS,
    KernelSpec('Flatten', _kernel_flatten, 'shape'),
    KernelSpec('Linear', _kernel_linear, 'linear'),
    KernelSpec('MaxPool2d', _kernel_maxpool2d, 'pool'),
    KernelSpec('AvgPool2d', _kernel_avgpool2d, 'pool'),
)


def make_default_registry() -> KernelRegistry:
    """Build a KernelRegistry with all Phase-1/2 reference kernels."""
    reg = KernelRegistry()
    for spec in DEFAULT_KERNEL_SPECS:
        reg.register(spec.op_name, spec.fn, category=spec.category)
    return reg
