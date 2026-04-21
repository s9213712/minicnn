"""Minimal training utilities for cuda_native.

Provides a single-step training function that wires together:
  forward with cache → loss → backward → SGD parameter update

This module is intentionally small.  Schedulers, weight decay, and
multi-epoch loops are out of scope for Phase 3.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from minicnn.cuda_native.backward import BackwardExecutor, make_default_backward_registry
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.loss import cross_entropy_loss, mse_loss


LossType = Literal['cross_entropy', 'mse']


def sgd_update(
    params: dict[str, np.ndarray],
    param_grads: dict[str, np.ndarray],
    lr: float,
    weight_decay: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return a new param dict after one SGD step.

    Does not mutate the input dicts.
    """
    updated: dict[str, np.ndarray] = {}
    for key, val in params.items():
        if key in param_grads:
            g = param_grads[key]
            if weight_decay > 0.0:
                g = g + weight_decay * val
            updated[key] = (val - lr * g).astype(np.float32)
        else:
            updated[key] = val
    return updated


def train_step(
    graph: NativeGraph,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    lr: float,
    loss_type: LossType = 'cross_entropy',
    weight_decay: float = 0.0,
    fwd_executor: ForwardExecutor | None = None,
    bwd_executor: BackwardExecutor | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    """Execute one forward-backward-update cycle.

    Args:
        graph:      NativeGraph built with build_graph().
        x:          Input batch, shape matching graph.input_spec.shape.
        y:          Labels — integer class indices (N,) for cross_entropy,
                    or float targets for mse.
        params:     Weight dict {'_w_{node}': ..., '_b_{node}': ...}.
        lr:         Learning rate for SGD.
        loss_type:  'cross_entropy' or 'mse'.
        weight_decay: L2 regularisation coefficient.
        fwd_executor: optional pre-built ForwardExecutor.
        bwd_executor: optional pre-built BackwardExecutor.

    Returns:
        (loss_value, updated_params)
    """
    if graph.input_spec is None:
        raise ValueError('Graph has no input_spec')

    fwd = fwd_executor or ForwardExecutor()
    bwd = bwd_executor or BackwardExecutor()

    # Forward pass — cache activations for backward
    ctx, cache = fwd.run_with_cache(
        graph,
        {graph.input_spec.name: x},
        params=params,
    )

    # Loss + initial gradient
    out_name = graph.output_spec.name if graph.output_spec else graph.nodes[-1].outputs[0]
    logits = ctx[out_name]

    if loss_type == 'cross_entropy':
        loss_val, grad_logits = cross_entropy_loss(logits, y)
    elif loss_type == 'mse':
        loss_val, grad_logits = mse_loss(logits, y.astype(np.float32))
    else:
        raise ValueError(f'Unknown loss_type: {loss_type!r}. Choose "cross_entropy" or "mse".')

    # Backward pass
    _grad_input, param_grads = bwd.run(graph, grad_logits, cache)

    # SGD update
    updated_params = sgd_update(params, param_grads, lr, weight_decay)

    return loss_val, updated_params
