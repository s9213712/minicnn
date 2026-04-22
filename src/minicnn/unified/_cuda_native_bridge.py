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
        if node.op_type == 'Conv2d':
            s = node.input_specs[0].shape if node.input_specs else None
            out_spec = node.output_specs[0] if node.output_specs else None
            if s is None or out_spec is None:
                continue
            c_in = s[1]
            c_out = out_spec.shape[1]
            kh = kw = node.attrs.get('kernel_size', 3)
            fan_in = int(c_in * kh * kw)
            w = rng.standard_normal((c_out, c_in, kh, kw)) * np.sqrt(2.0 / fan_in)
            params[f'_w_{node.name}'] = w.astype(np.float32)

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

    return params
