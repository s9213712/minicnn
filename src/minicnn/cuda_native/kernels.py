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


def _gelu_approx(x: np.ndarray) -> np.ndarray:
    return (
        0.5
        * x
        * (
            1.0
            + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
            )
        )
    ).astype(np.float32)


def _conv2d_forward_array(
    x: np.ndarray,
    w: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    stride: int | tuple[int, int] | list[int] = 1,
    padding: int | tuple[int, int] | list[int] = 0,
    groups: int = 1,
    node_desc: str = 'Conv2d',
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    sh, sw = _attr_pair(stride, label='stride', node=Node(name=node_desc, op_type='Conv2d'))
    ph, pw = _attr_pair(padding, label='padding', node=Node(name=node_desc, op_type='Conv2d'))
    if x.ndim != 4:
        raise ValueError(f'{node_desc} expects 4-D input (N,C,H,W), got shape {x.shape}')
    if w.ndim != 4:
        raise ValueError(f'{node_desc}: weight must be 4-D, got shape {w.shape}')
    _n, c_in, _h_in, _w_in = x.shape
    if groups <= 0:
        raise ValueError(f'{node_desc}: groups must be positive, got {groups}')
    if c_in % groups != 0:
        raise ValueError(f'{node_desc}: input channels {c_in} must be divisible by groups={groups}')
    c_out, w_in_per_group, kh, kw = w.shape
    if c_out % groups != 0:
        raise ValueError(f'{node_desc}: output channels {c_out} must be divisible by groups={groups}')
    expected_w_in = c_in // groups
    if w_in_per_group != expected_w_in:
        raise ValueError(
            f'[E_CONV2D_CHANNEL_GROUP_MISMATCH] {node_desc}: '
            f'weight expects {w_in_per_group} channels per group, '
            f'got input channels={c_in}, groups={groups}'
        )
    if ph > 0 or pw > 0:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    windows = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw, :, :]
    if windows.shape[2] == 0 or windows.shape[3] == 0:
        raise ValueError(
            f'{node_desc}: invalid output shape for input={x.shape}, '
            f'kernel={(kh, kw)}, stride={(sh, sw)}, padding={(ph, pw)}'
        )
    h_out = windows.shape[2]
    w_out = windows.shape[3]
    out = np.zeros((x.shape[0], c_out, h_out, w_out), dtype=np.float32)
    in_per_group = c_in // groups
    out_per_group = c_out // groups
    for group_idx in range(groups):
        in_start = group_idx * in_per_group
        in_end = (group_idx + 1) * in_per_group
        out_start = group_idx * out_per_group
        out_end = (group_idx + 1) * out_per_group
        group_windows = windows[:, in_start:in_end, :, :, :, :]
        group_weights = w[out_start:out_end, :, :, :]
        out[:, out_start:out_end, :, :] = np.tensordot(
            group_windows,
            group_weights,
            axes=([1, 4, 5], [1, 2, 3]),
        ).transpose(0, 3, 1, 2)
    if bias is not None:
        b_arr = np.asarray(bias, dtype=np.float32)
        if b_arr.shape != (c_out,):
            raise ValueError(f'{node_desc}: bias must have shape {(c_out,)}, got {b_arr.shape}')
        out = out + b_arr[None, :, None, None]
    return out.astype(np.float32)


def _batchnorm2d_forward_array(
    x: np.ndarray,
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float,
    momentum: float,
    mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if x.ndim != 4:
        raise ValueError(f'BatchNorm2d expects 4-D input (N,C,H,W), got shape {x.shape}')
    if mode == 'train':
        mean = x.mean(axis=(0, 2, 3)).astype(np.float32)
        var = x.var(axis=(0, 2, 3)).astype(np.float32)
        next_mean = ((1.0 - momentum) * running_mean + momentum * mean).astype(np.float32)
        next_var = ((1.0 - momentum) * running_var + momentum * var).astype(np.float32)
    else:
        mean = running_mean.astype(np.float32)
        var = running_var.astype(np.float32)
        next_mean = running_mean.astype(np.float32)
        next_var = running_var.astype(np.float32)
    centered = x - mean[None, :, None, None]
    inv_std = (1.0 / np.sqrt(var[None, :, None, None] + eps)).astype(np.float32)
    x_hat = (centered * inv_std).astype(np.float32)
    out = (x_hat * gamma[None, :, None, None] + beta[None, :, None, None]).astype(np.float32)
    return out, {
        'mean': mean.astype(np.float32),
        'var': var.astype(np.float32),
        'x_hat': x_hat.astype(np.float32),
        'inv_std': inv_std.astype(np.float32),
        'running_mean': next_mean.astype(np.float32),
        'running_var': next_var.astype(np.float32),
        'mode': np.asarray(0 if mode == 'eval' else 1, dtype=np.int32),
    }


def _layernorm2d_forward_array(
    x: np.ndarray,
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if x.ndim != 4:
        raise ValueError(f'LayerNorm2d expects 4-D input (N,C,H,W), got shape {x.shape}')
    mean = x.mean(axis=1, keepdims=True).astype(np.float32)
    var = x.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    out = (x_hat * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)).astype(np.float32)
    return out, {
        'mean': mean.astype(np.float32),
        'var': var.astype(np.float32),
        'x_hat': x_hat.astype(np.float32),
        'inv_std': inv_std.astype(np.float32),
    }


def _groupnorm_forward_array(
    x: np.ndarray,
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    num_groups: int,
    eps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if x.ndim != 4:
        raise ValueError(f'GroupNorm expects 4-D input (N,C,H,W), got shape {x.shape}')
    n, c, h, w = x.shape
    if c % num_groups != 0:
        raise ValueError(
            f'GroupNorm expects num_groups={num_groups} to divide channels={c}'
        )
    channels_per_group = c // num_groups
    x_group = x.reshape(n, num_groups, channels_per_group, h, w)
    mean = x_group.mean(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    var = x_group.var(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat_group = ((x_group - mean) * inv_std).astype(np.float32)
    x_hat = x_hat_group.reshape(n, c, h, w).astype(np.float32)
    out = (x_hat * gamma.reshape(1, c, 1, 1) + beta.reshape(1, c, 1, 1)).astype(np.float32)
    return out, {
        'x_hat': x_hat.astype(np.float32),
        'mean': mean.astype(np.float32),
        'var': var.astype(np.float32),
        'inv_std': inv_std.astype(np.float32),
        'num_groups': np.asarray(num_groups, dtype=np.int32),
    }


def _layernorm_forward_array(
    x: np.ndarray,
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not normalized_shape:
        raise ValueError('LayerNorm expects non-empty normalized_shape')
    if x.ndim < len(normalized_shape):
        raise ValueError(
            f'LayerNorm expects input rank >= {len(normalized_shape)}, got shape {x.shape}'
        )
    if tuple(int(v) for v in x.shape[-len(normalized_shape):]) != tuple(normalized_shape):
        raise ValueError(
            f'LayerNorm expects trailing shape {tuple(normalized_shape)}, got {tuple(x.shape[-len(normalized_shape):])}'
        )
    reduce_axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    mean = x.mean(axis=reduce_axes, keepdims=True).astype(np.float32)
    var = x.var(axis=reduce_axes, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    reshape = (1,) * (x.ndim - len(normalized_shape)) + tuple(normalized_shape)
    out = (x_hat * gamma.reshape(reshape) + beta.reshape(reshape)).astype(np.float32)
    return out, {
        'mean': mean.astype(np.float32),
        'var': var.astype(np.float32),
        'x_hat': x_hat.astype(np.float32),
        'inv_std': inv_std.astype(np.float32),
    }


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

    groups = int(node.attrs.get('groups', x.shape[1] if node.op_type in {'DepthwiseConv2d', 'depthwise_conv2d'} else 1))
    ctx[node.outputs[0]] = _conv2d_forward_array(
        x,
        w,
        bias=None if b is None else np.asarray(b, dtype=np.float32),
        stride=(sh, sw),
        padding=(ph, pw),
        groups=groups,
        node_desc=f'Conv2d node={node.name}',
    )


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

    out, cache = _batchnorm2d_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
        momentum=float(node.attrs.get('momentum', 0.1)),
        mode=mode,
    )
    next_mean = cache['running_mean']
    next_var = cache['running_var']
    if rm_key in ctx:
        ctx[rm_key][...] = next_mean
    else:
        ctx[rm_key] = next_mean
    if rv_key in ctx:
        ctx[rv_key][...] = next_var
    else:
        ctx[rv_key] = next_var
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


def _kernel_gelu(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_elementwise(
        node,
        ctx,
        transform=lambda x: 0.5 * x * (
            1.0
            + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
            )
        ),
    )


def _kernel_identity(node: Node, ctx: dict[str, Any]) -> None:
    _kernel_elementwise(node, ctx, transform=lambda x: x)


def _kernel_add(node: Node, ctx: dict[str, Any]) -> None:
    if len(node.inputs) < 2:
        raise ValueError(
            f'Add node={node.name}: expected at least two input tensors, got {len(node.inputs)}'
        )
    arrays = [np.asarray(ctx[input_name], dtype=np.float32) for input_name in node.inputs]
    ref_shape = arrays[0].shape
    result = arrays[0].copy()
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.shape != ref_shape:
            raise ValueError(
                f'Add node={node.name}: all inputs must share the same shape, '
                f'got input[0]={ref_shape} and input[{idx}]={arr.shape}'
            )
        result += arr
    ctx[node.outputs[0]] = result.astype(np.float32)


def _kernel_concat(node: Node, ctx: dict[str, Any]) -> None:
    if len(node.inputs) < 2:
        raise ValueError(
            f'Concat node={node.name}: expected at least two input tensors, got {len(node.inputs)}'
        )
    arrays = [np.asarray(ctx[input_name], dtype=np.float32) for input_name in node.inputs]
    axis = int(node.attrs.get('axis', 1))
    try:
        result = np.concatenate(arrays, axis=axis)
    except ValueError as exc:
        raise ValueError(f'Concat node={node.name}: {exc}') from exc
    ctx[node.outputs[0]] = result.astype(np.float32)


def _kernel_layernorm2d(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(
            f'LayerNorm2d expects 4-D input (N,C,H,W), got shape {x.shape}'
        )
    channels = x.shape[1]
    eps = float(node.attrs.get('eps', 1e-6))
    gamma = np.asarray(
        ctx.get(f'_w_{node.name}', np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    beta = np.asarray(
        ctx.get(f'_b_{node.name}', np.zeros(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    if gamma.shape != (channels,):
        raise ValueError(
            f'LayerNorm2d node={node.name}: weight must have shape {(channels,)}, got {gamma.shape}'
        )
    if beta.shape != (channels,):
        raise ValueError(
            f'LayerNorm2d node={node.name}: bias must have shape {(channels,)}, got {beta.shape}'
        )
    y, _cache = _layernorm2d_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        eps=eps,
    )
    ctx[node.outputs[0]] = y.astype(np.float32)


def _kernel_groupnorm(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(
            f'GroupNorm expects 4-D input (N,C,H,W), got shape {x.shape}'
        )
    channels = x.shape[1]
    num_groups = int(node.attrs.get('num_groups', 0))
    if num_groups <= 0:
        raise ValueError(
            f'GroupNorm node={node.name}: attr "num_groups" must be > 0, got {num_groups}'
        )
    eps = float(node.attrs.get('eps', 1e-5))
    gamma = np.asarray(
        ctx.get(f'_w_{node.name}', np.ones(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    beta = np.asarray(
        ctx.get(f'_b_{node.name}', np.zeros(channels, dtype=np.float32)),
        dtype=np.float32,
    )
    if gamma.shape != (channels,):
        raise ValueError(
            f'GroupNorm node={node.name}: weight must have shape {(channels,)}, got {gamma.shape}'
        )
    if beta.shape != (channels,):
        raise ValueError(
            f'GroupNorm node={node.name}: bias must have shape {(channels,)}, got {beta.shape}'
        )
    y, _cache = _groupnorm_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        num_groups=num_groups,
        eps=eps,
    )
    ctx[node.outputs[0]] = y.astype(np.float32)


def _kernel_layernorm(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    raw_shape = node.attrs.get('normalized_shape')
    if raw_shape is None:
        raise ValueError(f'LayerNorm node={node.name}: missing required attr "normalized_shape"')
    normalized_shape = (
        (int(raw_shape),)
        if isinstance(raw_shape, int)
        else tuple(int(v) for v in raw_shape)
    )
    eps = float(node.attrs.get('eps', 1e-5))
    gamma = np.asarray(
        ctx.get(f'_w_{node.name}', np.ones(normalized_shape, dtype=np.float32)),
        dtype=np.float32,
    )
    beta = np.asarray(
        ctx.get(f'_b_{node.name}', np.zeros(normalized_shape, dtype=np.float32)),
        dtype=np.float32,
    )
    if gamma.shape != normalized_shape:
        raise ValueError(
            f'LayerNorm node={node.name}: weight must have shape {normalized_shape}, got {gamma.shape}'
        )
    if beta.shape != normalized_shape:
        raise ValueError(
            f'LayerNorm node={node.name}: bias must have shape {normalized_shape}, got {beta.shape}'
        )
    y, _cache = _layernorm_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        normalized_shape=normalized_shape,
        eps=eps,
    )
    ctx[node.outputs[0]] = y.astype(np.float32)


def _kernel_global_avgpool2d(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(
            f'{node.op_type} expects 4-D input (N,C,H,W), got shape {x.shape}'
        )
    ctx[node.outputs[0]] = x.mean(axis=(2, 3), keepdims=True).astype(np.float32)


def _kernel_adaptive_avgpool2d(node: Node, ctx: dict[str, Any]) -> None:
    output_size = node.attrs.get('output_size', 1)
    normalized = tuple(output_size) if isinstance(output_size, (list, tuple)) else output_size
    if normalized not in {1, (1, 1)}:
        raise ValueError(
            f'AdaptiveAvgPool2d node={node.name}: only output_size=1 or (1, 1) is supported, got {output_size!r}'
        )
    _kernel_global_avgpool2d(node, ctx)


def _kernel_dropout(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    p = float(node.attrs.get('p', 0.5))
    mode = str(ctx.get('__mode__', 'eval'))
    if mode == 'eval' or p <= 0.0:
        ctx[node.outputs[0]] = x.astype(np.float32)
        ctx[f'__cache_{node.name}'] = {'mask': np.ones_like(x, dtype=np.float32)}
        return
    if p >= 1.0:
        raise ValueError(f'Dropout node={node.name}: p must be < 1.0, got {p}')
    rng = ctx.get('__dropout_rng__')
    if rng is None:
        rng = np.random.default_rng(42)
        ctx['__dropout_rng__'] = rng
    keep_prob = 1.0 - p
    mask = (rng.random(x.shape, dtype=np.float32) < keep_prob).astype(np.float32) / keep_prob
    ctx[node.outputs[0]] = (x * mask).astype(np.float32)
    ctx[f'__cache_{node.name}'] = {'mask': mask.astype(np.float32)}


def _kernel_droppath(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    p = float(node.attrs.get('p', 0.0))
    mode = str(ctx.get('__mode__', 'eval'))
    if mode == 'eval' or p <= 0.0:
        ctx[node.outputs[0]] = x.astype(np.float32)
        ctx[f'__cache_{node.name}'] = {'mask': np.ones_like(x, dtype=np.float32)}
        return
    if p >= 1.0:
        raise ValueError(f'DropPath node={node.name}: p must be < 1.0, got {p}')
    rng = ctx.get('__dropout_rng__')
    if rng is None:
        rng = np.random.default_rng(42)
        ctx['__dropout_rng__'] = rng
    keep_prob = 1.0 - p
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (rng.random(mask_shape, dtype=np.float32) < keep_prob).astype(np.float32) / keep_prob
    mask = np.broadcast_to(mask, x.shape).astype(np.float32)
    ctx[node.outputs[0]] = (x * mask).astype(np.float32)
    ctx[f'__cache_{node.name}'] = {'mask': mask.astype(np.float32)}


def _kernel_residual_block(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    stride = int(node.attrs.get('stride', 1))
    kernel_size = int(node.attrs.get('kernel_size', 3))
    padding = int(node.attrs.get('padding', kernel_size // 2))
    bias = bool(node.attrs.get('bias', False))
    mode = str(ctx.get('__mode__', 'eval'))

    conv1 = _conv2d_forward_array(
        x,
        np.asarray(ctx[f'_w_conv1_{node.name}'], dtype=np.float32),
        bias=np.asarray(ctx[f'_b_conv1_{node.name}'], dtype=np.float32) if bias and f'_b_conv1_{node.name}' in ctx else None,
        stride=stride,
        padding=padding,
        groups=1,
        node_desc=f'ResidualBlock.conv1 node={node.name}',
    )
    bn1, bn1_cache = _batchnorm2d_forward_array(
        conv1,
        gamma=np.asarray(ctx[f'_w_bn1_{node.name}'], dtype=np.float32),
        beta=np.asarray(ctx[f'_b_bn1_{node.name}'], dtype=np.float32),
        running_mean=np.asarray(ctx[f'_running_mean_bn1_{node.name}'], dtype=np.float32),
        running_var=np.asarray(ctx[f'_running_var_bn1_{node.name}'], dtype=np.float32),
        eps=float(node.attrs.get('eps', 1e-5)),
        momentum=float(node.attrs.get('momentum', 0.1)),
        mode=mode,
    )
    ctx[f'_running_mean_bn1_{node.name}'][...] = bn1_cache['running_mean']
    ctx[f'_running_var_bn1_{node.name}'][...] = bn1_cache['running_var']
    act1 = np.maximum(bn1, 0.0).astype(np.float32)

    conv2 = _conv2d_forward_array(
        act1,
        np.asarray(ctx[f'_w_conv2_{node.name}'], dtype=np.float32),
        bias=np.asarray(ctx[f'_b_conv2_{node.name}'], dtype=np.float32) if bias and f'_b_conv2_{node.name}' in ctx else None,
        stride=1,
        padding=padding,
        groups=1,
        node_desc=f'ResidualBlock.conv2 node={node.name}',
    )
    bn2, bn2_cache = _batchnorm2d_forward_array(
        conv2,
        gamma=np.asarray(ctx[f'_w_bn2_{node.name}'], dtype=np.float32),
        beta=np.asarray(ctx[f'_b_bn2_{node.name}'], dtype=np.float32),
        running_mean=np.asarray(ctx[f'_running_mean_bn2_{node.name}'], dtype=np.float32),
        running_var=np.asarray(ctx[f'_running_var_bn2_{node.name}'], dtype=np.float32),
        eps=float(node.attrs.get('eps', 1e-5)),
        momentum=float(node.attrs.get('momentum', 0.1)),
        mode=mode,
    )
    ctx[f'_running_mean_bn2_{node.name}'][...] = bn2_cache['running_mean']
    ctx[f'_running_var_bn2_{node.name}'][...] = bn2_cache['running_var']

    use_shortcut_conv = bool(ctx.get(f'_w_shortcut_conv_{node.name}') is not None) if f'_w_shortcut_conv_{node.name}' in ctx else False
    if use_shortcut_conv:
        shortcut_conv = _conv2d_forward_array(
            x,
            np.asarray(ctx[f'_w_shortcut_conv_{node.name}'], dtype=np.float32),
            bias=None,
            stride=stride,
            padding=0,
            groups=1,
            node_desc=f'ResidualBlock.shortcut_conv node={node.name}',
        )
        shortcut, shortcut_cache = _batchnorm2d_forward_array(
            shortcut_conv,
            gamma=np.asarray(ctx[f'_w_shortcut_bn_{node.name}'], dtype=np.float32),
            beta=np.asarray(ctx[f'_b_shortcut_bn_{node.name}'], dtype=np.float32),
            running_mean=np.asarray(ctx[f'_running_mean_shortcut_bn_{node.name}'], dtype=np.float32),
            running_var=np.asarray(ctx[f'_running_var_shortcut_bn_{node.name}'], dtype=np.float32),
            eps=float(node.attrs.get('eps', 1e-5)),
            momentum=float(node.attrs.get('momentum', 0.1)),
            mode=mode,
        )
        ctx[f'_running_mean_shortcut_bn_{node.name}'][...] = shortcut_cache['running_mean']
        ctx[f'_running_var_shortcut_bn_{node.name}'][...] = shortcut_cache['running_var']
    else:
        shortcut = x.astype(np.float32)
        shortcut_conv = None
        shortcut_cache = None

    summed = (bn2 + shortcut).astype(np.float32)
    out = np.maximum(summed, 0.0).astype(np.float32)
    ctx[node.outputs[0]] = out
    ctx[f'__cache_{node.name}'] = {
        'x': x.astype(np.float32),
        'conv1': conv1.astype(np.float32),
        'bn1': bn1.astype(np.float32),
        'bn1_cache': bn1_cache,
        'act1': act1.astype(np.float32),
        'conv2': conv2.astype(np.float32),
        'bn2_cache': bn2_cache,
        'shortcut': shortcut.astype(np.float32),
        'shortcut_conv': None if shortcut_conv is None else shortcut_conv.astype(np.float32),
        'shortcut_cache': shortcut_cache,
        'summed': summed.astype(np.float32),
        'use_shortcut_conv': np.asarray(1 if use_shortcut_conv else 0, dtype=np.int32),
    }


def _kernel_convnext_block(node: Node, ctx: dict[str, Any]) -> None:
    x = np.asarray(ctx[node.inputs[0]], dtype=np.float32)
    kernel_size = int(node.attrs.get('kernel_size', 7))
    padding = kernel_size // 2
    bias = bool(node.attrs.get('bias', True))
    depthwise = _conv2d_forward_array(
        x,
        np.asarray(ctx[f'_w_depthwise_{node.name}'], dtype=np.float32),
        bias=np.asarray(ctx[f'_b_depthwise_{node.name}'], dtype=np.float32) if bias and f'_b_depthwise_{node.name}' in ctx else None,
        stride=1,
        padding=padding,
        groups=x.shape[1],
        node_desc=f'ConvNeXtBlock.depthwise node={node.name}',
    )
    norm, norm_cache = _layernorm2d_forward_array(
        depthwise,
        gamma=np.asarray(ctx[f'_w_ln_{node.name}'], dtype=np.float32),
        beta=np.asarray(ctx[f'_b_ln_{node.name}'], dtype=np.float32),
        eps=float(node.attrs.get('layer_norm_eps', 1e-6)),
    )
    pw1 = _conv2d_forward_array(
        norm,
        np.asarray(ctx[f'_w_pw1_{node.name}'], dtype=np.float32),
        bias=np.asarray(ctx[f'_b_pw1_{node.name}'], dtype=np.float32) if bias and f'_b_pw1_{node.name}' in ctx else None,
        stride=1,
        padding=0,
        groups=1,
        node_desc=f'ConvNeXtBlock.pointwise1 node={node.name}',
    )
    activated = _gelu_approx(pw1)
    pw2 = _conv2d_forward_array(
        activated,
        np.asarray(ctx[f'_w_pw2_{node.name}'], dtype=np.float32),
        bias=np.asarray(ctx[f'_b_pw2_{node.name}'], dtype=np.float32) if bias and f'_b_pw2_{node.name}' in ctx else None,
        stride=1,
        padding=0,
        groups=1,
        node_desc=f'ConvNeXtBlock.pointwise2 node={node.name}',
    )
    if f'_layer_scale_{node.name}' in ctx:
        layer_scale = np.asarray(ctx[f'_layer_scale_{node.name}'], dtype=np.float32).reshape(1, -1, 1, 1)
        scaled = (pw2 * layer_scale).astype(np.float32)
    else:
        layer_scale = None
        scaled = pw2.astype(np.float32)
    out = (x + scaled).astype(np.float32)
    ctx[node.outputs[0]] = out
    ctx[f'__cache_{node.name}'] = {
        'x': x.astype(np.float32),
        'depthwise': depthwise.astype(np.float32),
        'norm_cache': norm_cache,
        'norm': norm.astype(np.float32),
        'pw1': pw1.astype(np.float32),
        'activated': activated.astype(np.float32),
        'pw2': pw2.astype(np.float32),
        'scaled': scaled.astype(np.float32),
        'has_layer_scale': np.asarray(1 if layer_scale is not None else 0, dtype=np.int32),
    }


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
    KernelSpec('GELU', _kernel_gelu, 'activation'),
)


DEFAULT_KERNEL_SPECS: tuple[KernelSpec, ...] = (
    KernelSpec('Add', _kernel_add, 'elementwise'),
    KernelSpec('Concat', _kernel_concat, 'elementwise'),
    KernelSpec('Conv2d', _kernel_conv2d, 'convolution'),
    KernelSpec('DepthwiseConv2d', _kernel_conv2d, 'convolution'),
    KernelSpec('depthwise_conv2d', _kernel_conv2d, 'convolution'),
    KernelSpec('PointwiseConv2d', _kernel_conv2d, 'convolution'),
    KernelSpec('pointwise_conv2d', _kernel_conv2d, 'convolution'),
    KernelSpec('ResidualBlock', _kernel_residual_block, 'composite'),
    KernelSpec('ConvNeXtBlock', _kernel_convnext_block, 'composite'),
    KernelSpec('convnext_block', _kernel_convnext_block, 'composite'),
    KernelSpec('BatchNorm2d', _kernel_batchnorm2d_eval, 'normalization'),
    KernelSpec('GroupNorm', _kernel_groupnorm, 'normalization'),
    KernelSpec('LayerNorm', _kernel_layernorm, 'normalization'),
    KernelSpec('LayerNorm2d', _kernel_layernorm2d, 'normalization'),
    KernelSpec('layernorm2d', _kernel_layernorm2d, 'normalization'),
    *_ACTIVATION_KERNEL_SPECS,
    KernelSpec('Identity', _kernel_identity, 'activation'),
    KernelSpec('Dropout', _kernel_dropout, 'regularization'),
    KernelSpec('DropPath', _kernel_droppath, 'regularization'),
    KernelSpec('Flatten', _kernel_flatten, 'shape'),
    KernelSpec('Linear', _kernel_linear, 'linear'),
    KernelSpec('MaxPool2d', _kernel_maxpool2d, 'pool'),
    KernelSpec('AvgPool2d', _kernel_avgpool2d, 'pool'),
    KernelSpec('AdaptiveAvgPool2d', _kernel_adaptive_avgpool2d, 'pool'),
    KernelSpec('GlobalAvgPool2d', _kernel_global_avgpool2d, 'pool'),
)


def make_default_registry() -> KernelRegistry:
    """Build a KernelRegistry with all Phase-1/2 reference kernels."""
    reg = KernelRegistry()
    for spec in DEFAULT_KERNEL_SPECS:
        reg.register(spec.op_name, spec.fn, category=spec.category)
    return reg
