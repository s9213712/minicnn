from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.graph import NativeGraph
from minicnn.unified._cuda_native_support import (
    best_checkpoint_path as _best_checkpoint_path,
    build_epoch_row as _build_epoch_row,
    build_training_summary as _build_training_summary,
    epoch_log_message as _epoch_log_message,
    evaluate_native_graph as _evaluate,
    load_numpy_data as _load_numpy_data,
    make_scheduler as _make_scheduler,
    resolve_loss_type as _resolve_loss_type,
)


def _init_params(graph: NativeGraph, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}

    for node in graph.nodes:
        if node.op_type in {'Conv2d', 'DepthwiseConv2d', 'depthwise_conv2d', 'PointwiseConv2d', 'pointwise_conv2d'}:
            s = node.input_specs[0].shape if node.input_specs else None
            out_spec = node.output_specs[0] if node.output_specs else None
            if s is None or out_spec is None:
                continue
            c_in = s[1]
            c_out = out_spec.shape[1]
            kernel_size = node.attrs.get(
                'kernel_size',
                1 if node.op_type in {'PointwiseConv2d', 'pointwise_conv2d'} else 3,
            )
            if isinstance(kernel_size, (tuple, list)):
                kh, kw = int(kernel_size[0]), int(kernel_size[1])
            else:
                kh = kw = int(kernel_size)
            groups = int(
                node.attrs.get(
                    'groups',
                    c_in if node.op_type in {'DepthwiseConv2d', 'depthwise_conv2d'} else 1,
                )
            )
            fan_in = int((c_in // groups) * kh * kw)
            w = rng.standard_normal((c_out, c_in // groups, kh, kw)) * np.sqrt(2.0 / fan_in)
            params[f'_w_{node.name}'] = w.astype(np.float32)
            if bool(node.attrs.get('bias', True)):
                params[f'_b_{node.name}'] = np.zeros(c_out, dtype=np.float32)

        elif node.op_type == 'Linear':
            s = node.input_specs[0].shape if node.input_specs else None
            out_spec = node.output_specs[0] if node.output_specs else None
            if s is None or out_spec is None:
                continue
            in_f = s[1]
            out_f = out_spec.shape[1]
            w = rng.standard_normal((out_f, in_f)) * np.sqrt(2.0 / in_f)
            params[f'_w_{node.name}'] = w.astype(np.float32)
            params[f'_b_{node.name}'] = np.zeros(out_f, dtype=np.float32)
        elif node.op_type == 'BatchNorm2d':
            s = node.input_specs[0].shape if node.input_specs else None
            if s is None:
                continue
            channels = int(s[1])
            params[f'_w_{node.name}'] = np.ones(channels, dtype=np.float32)
            params[f'_b_{node.name}'] = np.zeros(channels, dtype=np.float32)
            params[f'_running_mean_{node.name}'] = np.zeros(channels, dtype=np.float32)
            params[f'_running_var_{node.name}'] = np.ones(channels, dtype=np.float32)
        elif node.op_type == 'GroupNorm':
            s = node.input_specs[0].shape if node.input_specs else None
            if s is None:
                continue
            channels = int(s[1])
            params[f'_w_{node.name}'] = np.ones(channels, dtype=np.float32)
            params[f'_b_{node.name}'] = np.zeros(channels, dtype=np.float32)
        elif node.op_type == 'LayerNorm':
            raw_shape = node.attrs.get('normalized_shape')
            if raw_shape is None:
                continue
            normalized_shape = (
                (int(raw_shape),)
                if isinstance(raw_shape, int)
                else tuple(int(v) for v in raw_shape)
            )
            params[f'_w_{node.name}'] = np.ones(normalized_shape, dtype=np.float32)
            params[f'_b_{node.name}'] = np.zeros(normalized_shape, dtype=np.float32)
        elif node.op_type in {'LayerNorm2d', 'layernorm2d'}:
            s = node.input_specs[0].shape if node.input_specs else None
            if s is None:
                continue
            channels = int(s[1])
            params[f'_w_{node.name}'] = np.ones(channels, dtype=np.float32)
            params[f'_b_{node.name}'] = np.zeros(channels, dtype=np.float32)
        elif node.op_type == 'ResidualBlock':
            s = node.input_specs[0].shape if node.input_specs else None
            out_spec = node.output_specs[0] if node.output_specs else None
            if s is None or out_spec is None:
                continue
            c_in = int(s[1])
            c_out = int(out_spec.shape[1])
            kernel_size = int(node.attrs.get('kernel_size', 3))
            bias = bool(node.attrs.get('bias', False))
            fan_in1 = int(c_in * kernel_size * kernel_size)
            fan_in2 = int(c_out * kernel_size * kernel_size)
            w1 = rng.standard_normal((c_out, c_in, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in1)
            w2 = rng.standard_normal((c_out, c_out, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in2)
            params[f'_w_conv1_{node.name}'] = w1.astype(np.float32)
            params[f'_w_conv2_{node.name}'] = w2.astype(np.float32)
            if bias:
                params[f'_b_conv1_{node.name}'] = np.zeros(c_out, dtype=np.float32)
                params[f'_b_conv2_{node.name}'] = np.zeros(c_out, dtype=np.float32)
            for suffix in ('bn1', 'bn2'):
                params[f'_w_{suffix}_{node.name}'] = np.ones(c_out, dtype=np.float32)
                params[f'_b_{suffix}_{node.name}'] = np.zeros(c_out, dtype=np.float32)
                params[f'_running_mean_{suffix}_{node.name}'] = np.zeros(c_out, dtype=np.float32)
                params[f'_running_var_{suffix}_{node.name}'] = np.ones(c_out, dtype=np.float32)
            stride = int(node.attrs.get('stride', 1))
            if stride != 1 or c_in != c_out:
                fan_in_short = int(c_in)
                w_short = rng.standard_normal((c_out, c_in, 1, 1)) * np.sqrt(2.0 / fan_in_short)
                params[f'_w_shortcut_conv_{node.name}'] = w_short.astype(np.float32)
                params[f'_w_shortcut_bn_{node.name}'] = np.ones(c_out, dtype=np.float32)
                params[f'_b_shortcut_bn_{node.name}'] = np.zeros(c_out, dtype=np.float32)
                params[f'_running_mean_shortcut_bn_{node.name}'] = np.zeros(c_out, dtype=np.float32)
                params[f'_running_var_shortcut_bn_{node.name}'] = np.ones(c_out, dtype=np.float32)
        elif node.op_type in {'ConvNeXtBlock', 'convnext_block'}:
            s = node.input_specs[0].shape if node.input_specs else None
            if s is None:
                continue
            channels = int(s[1])
            kernel_size = int(node.attrs.get('kernel_size', 7))
            hidden_channels = int(node.attrs.get('hidden_channels', round(channels * float(node.attrs.get('expansion_ratio', 4.0)))))
            bias = bool(node.attrs.get('bias', True))
            depthwise = rng.standard_normal((channels, 1, kernel_size, kernel_size)) * np.sqrt(2.0 / (kernel_size * kernel_size))
            pw1 = rng.standard_normal((hidden_channels, channels, 1, 1)) * np.sqrt(2.0 / channels)
            pw2 = rng.standard_normal((channels, hidden_channels, 1, 1)) * np.sqrt(2.0 / hidden_channels)
            params[f'_w_depthwise_{node.name}'] = depthwise.astype(np.float32)
            params[f'_w_ln_{node.name}'] = np.ones(channels, dtype=np.float32)
            params[f'_b_ln_{node.name}'] = np.zeros(channels, dtype=np.float32)
            params[f'_w_pw1_{node.name}'] = pw1.astype(np.float32)
            params[f'_w_pw2_{node.name}'] = pw2.astype(np.float32)
            if bias:
                params[f'_b_depthwise_{node.name}'] = np.zeros(channels, dtype=np.float32)
                params[f'_b_pw1_{node.name}'] = np.zeros(hidden_channels, dtype=np.float32)
                params[f'_b_pw2_{node.name}'] = np.zeros(channels, dtype=np.float32)
            layer_scale_init_value = float(node.attrs.get('layer_scale_init_value', 1e-6))
            if layer_scale_init_value > 0:
                params[f'_layer_scale_{node.name}'] = np.full(channels, layer_scale_init_value, dtype=np.float32)

    return params
