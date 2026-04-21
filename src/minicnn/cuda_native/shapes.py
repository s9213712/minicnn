"""Shape inference for cuda_native ops.

All shape inference is centralised here — executors and planners read
from TensorSpec.shape, never re-derive shapes themselves.
"""
from __future__ import annotations

from typing import Any


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def infer_conv2d(
    input_shape: tuple[int, ...],
    out_channels: int,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> tuple[int, ...]:
    """Return (N, C_out, H_out, W_out) for a Conv2d op."""
    if len(input_shape) != 4:
        raise ValueError(
            f'Conv2d expects 4-D input (N,C,H,W), got shape {input_shape}'
        )
    n, _c, h, w = input_shape
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(
            f'Invalid Conv2d output shape: ({oh}, {ow}). '
            f'input=({h},{w}), kernel=({kh},{kw}), stride=({sh},{sw}), padding=({ph},{pw})'
        )
    return (n, out_channels, oh, ow)


def infer_activation(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """ReLU, LeakyReLU — shape is unchanged."""
    return input_shape


def infer_flatten(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Flatten all dims except batch into one vector: (N, C*H*W)."""
    if len(input_shape) < 2:
        raise ValueError(
            f'Flatten expects at least 2-D input, got shape {input_shape}'
        )
    n = input_shape[0]
    flat = 1
    for d in input_shape[1:]:
        flat *= d
    return (n, flat)


def infer_linear(
    input_shape: tuple[int, ...],
    out_features: int,
) -> tuple[int, ...]:
    """Linear (dense) layer: (N, in_features) -> (N, out_features)."""
    if len(input_shape) != 2:
        raise ValueError(
            f'Linear expects 2-D input (N, in_features), got shape {input_shape}. '
            'Insert a Flatten layer before Linear.'
        )
    return (input_shape[0], out_features)


def infer_shape(
    op_type: str,
    input_shape: tuple[int, ...],
    attrs: dict[str, Any],
    node_name: str = '',
) -> tuple[int, ...]:
    """Dispatch to the correct shape-inference function for *op_type*.

    Raises ValueError with a node-level message on invalid configs.
    """
    loc = f' (node={node_name})' if node_name else ''
    try:
        if op_type == 'Conv2d':
            out_ch = attrs.get('out_channels')
            if out_ch is None:
                raise ValueError(f'Conv2d{loc}: missing required attr "out_channels"')
            return infer_conv2d(
                input_shape,
                out_channels=int(out_ch),
                kernel_size=attrs.get('kernel_size', 3),
                stride=attrs.get('stride', 1),
                padding=attrs.get('padding', 0),
            )
        if op_type in ('ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'SiLU'):
            return infer_activation(input_shape)
        if op_type == 'Flatten':
            return infer_flatten(input_shape)
        if op_type == 'Linear':
            out_f = attrs.get('out_features')
            if out_f is None:
                raise ValueError(f'Linear{loc}: missing required attr "out_features"')
            return infer_linear(input_shape, int(out_f))
        raise ValueError(f'No shape inference rule for op: {op_type}{loc}')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'Shape inference failed for {op_type}{loc}: {exc}') from exc
