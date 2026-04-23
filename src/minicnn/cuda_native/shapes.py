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


def infer_batchnorm2d(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """BatchNorm2d preserves shape but requires NCHW input."""
    if len(input_shape) != 4:
        raise ValueError(
            f'BatchNorm2d expects 4-D input (N,C,H,W), got shape {input_shape}'
        )
    return input_shape


def infer_layernorm2d(
    input_shape: tuple[int, ...],
    num_channels: int | None = None,
) -> tuple[int, ...]:
    """LayerNorm2d preserves NCHW shape and optionally validates channels."""
    if len(input_shape) != 4:
        raise ValueError(
            f'LayerNorm2d expects 4-D input (N,C,H,W), got shape {input_shape}'
        )
    if num_channels is not None and int(num_channels) != int(input_shape[1]):
        raise ValueError(
            f'LayerNorm2d num_channels={num_channels} does not match input channels={input_shape[1]}'
        )
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


def infer_pool2d(
    input_shape: tuple[int, ...],
    kernel_size: int | tuple[int, int] = 2,
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> tuple[int, ...]:
    """Return (N, C, H_out, W_out) for MaxPool2d or AvgPool2d."""
    if len(input_shape) != 4:
        raise ValueError(
            f'Pool2d expects 4-D input (N,C,H,W), got shape {input_shape}'
        )
    n, c, h, w = input_shape
    kh, kw = _pair(kernel_size)
    if kh <= 0 or kw <= 0:
        raise ValueError(f'Pool2d kernel_size must be positive, got {(kh, kw)}')
    if stride is None:
        sh, sw = kh, kw
    else:
        sh, sw = _pair(stride)
    if sh <= 0 or sw <= 0:
        raise ValueError(f'Pool2d stride must be positive, got {(sh, sw)}')
    ph, pw = _pair(padding)
    if ph < 0 or pw < 0:
        raise ValueError(f'Pool2d padding must be non-negative, got {(ph, pw)}')
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(
            f'Invalid Pool2d output shape: ({oh}, {ow}). '
            f'input=({h},{w}), kernel=({kh},{kw}), stride=({sh},{sw}), padding=({ph},{pw})'
        )
    return (n, c, oh, ow)


def infer_global_avgpool2d(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return (N, C, 1, 1) for channel-preserving global average pooling."""
    if len(input_shape) != 4:
        raise ValueError(
            f'GlobalAvgPool2d expects 4-D input (N,C,H,W), got shape {input_shape}'
        )
    n, c, _h, _w = input_shape
    return (n, c, 1, 1)


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
        if op_type in ('DepthwiseConv2d', 'depthwise_conv2d'):
            if len(input_shape) != 4:
                raise ValueError(
                    f'DepthwiseConv2d{loc}: expects 4-D input (N,C,H,W), got shape {input_shape}'
                )
            c_in = int(input_shape[1])
            channel_multiplier = int(attrs.get('channel_multiplier', 1))
            out_ch = attrs.get('out_channels')
            if out_ch is None:
                out_ch = c_in * channel_multiplier
            if int(out_ch) % c_in != 0:
                raise ValueError(
                    f'DepthwiseConv2d{loc}: out_channels={out_ch} must be a multiple of input channels={c_in}'
                )
            kernel_size = attrs.get('kernel_size', 3)
            if isinstance(kernel_size, (tuple, list)):
                default_padding = tuple(int(k) // 2 for k in kernel_size)
            else:
                default_padding = int(kernel_size) // 2
            return infer_conv2d(
                input_shape,
                out_channels=int(out_ch),
                kernel_size=kernel_size,
                stride=attrs.get('stride', 1),
                padding=attrs.get('padding', default_padding),
            )
        if op_type in ('PointwiseConv2d', 'pointwise_conv2d'):
            out_ch = attrs.get('out_channels')
            if out_ch is None:
                raise ValueError(f'PointwiseConv2d{loc}: missing required attr "out_channels"')
            return infer_conv2d(
                input_shape,
                out_channels=int(out_ch),
                kernel_size=1,
                stride=1,
                padding=0,
            )
        if op_type in ('ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'SiLU', 'GELU', 'Identity', 'Dropout'):
            return infer_activation(input_shape)
        if op_type == 'BatchNorm2d':
            return infer_batchnorm2d(input_shape)
        if op_type in ('LayerNorm2d', 'layernorm2d'):
            num_channels = attrs.get('num_channels', attrs.get('channels'))
            return infer_layernorm2d(input_shape, num_channels=num_channels)
        if op_type == 'ResidualBlock':
            out_channels = attrs.get('out_channels', attrs.get('channels', input_shape[1] if len(input_shape) == 4 else None))
            if out_channels is None:
                raise ValueError(f'ResidualBlock{loc}: missing channels/out_channels')
            kernel_size = int(attrs.get('kernel_size', 3))
            padding = attrs.get('padding', kernel_size // 2)
            return infer_conv2d(
                input_shape,
                out_channels=int(out_channels),
                kernel_size=kernel_size,
                stride=attrs.get('stride', 1),
                padding=padding,
            )
        if op_type in ('ConvNeXtBlock', 'convnext_block'):
            if len(input_shape) != 4:
                raise ValueError(
                    f'ConvNeXtBlock{loc}: expects 4-D input (N,C,H,W), got shape {input_shape}'
                )
            channels = attrs.get('channels', attrs.get('in_channels', input_shape[1]))
            if int(channels) != int(input_shape[1]):
                raise ValueError(
                    f'ConvNeXtBlock{loc}: channels={channels} must match input channels={input_shape[1]}'
                )
            return input_shape
        if op_type == 'Flatten':
            return infer_flatten(input_shape)
        if op_type == 'Linear':
            out_f = attrs.get('out_features')
            if out_f is None:
                raise ValueError(f'Linear{loc}: missing required attr "out_features"')
            return infer_linear(input_shape, int(out_f))
        if op_type in ('MaxPool2d', 'AvgPool2d'):
            return infer_pool2d(
                input_shape,
                kernel_size=attrs.get('kernel_size', 2),
                stride=attrs.get('stride', None),
                padding=attrs.get('padding', 0),
            )
        if op_type in ('GlobalAvgPool2d', 'AdaptiveAvgPool2d'):
            output_size = attrs.get('output_size', 1)
            normalized = tuple(output_size) if isinstance(output_size, (list, tuple)) else output_size
            if op_type == 'AdaptiveAvgPool2d' and normalized not in {1, (1, 1)}:
                raise ValueError(
                    f'AdaptiveAvgPool2d{loc}: only output_size=1 or (1, 1) is supported, got {output_size!r}'
                )
            return infer_global_avgpool2d(input_shape)
        raise ValueError(f'No shape inference rule for op: {op_type}{loc}')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'Shape inference failed for {op_type}{loc}: {exc}') from exc
