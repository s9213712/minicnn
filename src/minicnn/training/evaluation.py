"""Forward-pass evaluation helpers for the CUDA CNN."""
from __future__ import annotations

from ctypes import c_float

import numpy as np

from minicnn.core.cuda_backend import download_int_scalar, lib, zero_bytes
from minicnn.config.settings import BATCH, EVAL_MAX_BATCHES, get_arch
from minicnn.training.cuda_arch import CudaNetGeometry
from minicnn.training.cuda_ops import (
    cnhw_to_nchw_into,
    conv_forward_into,
    malloc_floats,
    malloc_ints,
    maxpool_forward_into,
    upload_int_to,
    upload_to,
)


def _free_ptrs(ptrs: list) -> None:
    for ptr in reversed(ptrs):
        if ptr:
            lib.gpu_free(ptr)


class EvalWorkspace:
    """Pre-allocated GPU buffers for the forward pass (no gradient storage)."""

    def __init__(self, batch_size: int) -> None:
        geom = get_arch()
        self.geom = geom
        self.batch_size = int(batch_size)
        n = self.batch_size
        self.ptrs: list = []

        s0 = geom.conv_stages[0]
        self.d_x = self._track(malloc_floats(n * s0.in_c * s0.h_in * s0.w_in))

        self.d_col: list       = []
        self.d_conv_raw: list  = []
        self.d_conv_nchw: list = []
        self.d_ln_out: list    = []
        self.d_bn_out: list    = []
        self.d_pool: list      = []
        self.d_max_idx: list   = []
        self.d_pool_nchw: list = []

        for s in geom.conv_stages:
            self.d_col.append(self._track(malloc_floats(s.in_c * s.kh * s.kw * n * s.h_out * s.w_out)))
            self.d_conv_raw.append(self._track(malloc_floats(s.out_c * n * s.h_out * s.w_out)))
            if s.layer_norm and not s.pool:
                self.d_ln_out.append(self._track(malloc_floats(n * s.out_c * s.h_out * s.w_out)))
            else:
                self.d_ln_out.append(None)
            if s.batch_norm:
                self.d_bn_out.append(self._track(malloc_floats(n * s.out_c * s.h_out * s.w_out)))
            else:
                self.d_bn_out.append(None)
            if s.pool:
                self.d_pool.append(self._track(malloc_floats(s.out_c * n * s.ph * s.pw)))
                self.d_max_idx.append(self._track(malloc_ints(s.out_c * n * s.ph * s.pw)))
                self.d_pool_nchw.append(self._track(malloc_floats(n * s.out_c * s.ph * s.pw)))
                self.d_conv_nchw.append(None)
            else:
                self.d_conv_nchw.append(self._track(malloc_floats(n * s.out_c * s.h_out * s.w_out)))
                self.d_pool.append(None)
                self.d_max_idx.append(None)
                self.d_pool_nchw.append(None)

        self.d_fc_out  = self._track(malloc_floats(n * geom.fc_out))
        self.d_y       = self._track(malloc_ints(n))
        self.d_correct = self._track(malloc_ints(1))

    def _track(self, ptr):
        self.ptrs.append(ptr)
        return ptr

    def free(self) -> None:
        _free_ptrs(self.ptrs)
        self.ptrs = []

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.free()
        except Exception as exc:
            import warnings
            warnings.warn(
                f"EvalWorkspace.__del__: GPU memory cleanup failed: {exc}",
                ResourceWarning,
                stacklevel=2,
            )


def _stage_output_ptr(i: int, ws: EvalWorkspace) -> object:
    """Return the NCHW output pointer of stage i."""
    s = ws.geom.conv_stages[i]
    if s.pool:
        return ws.d_pool_nchw[i]
    if s.batch_norm:
        return ws.d_bn_out[i]
    if s.layer_norm:
        return ws.d_ln_out[i]
    return ws.d_conv_nchw[i]


def _forward_logits_into(x: np.ndarray, device_weights, workspace: EvalWorkspace) -> object:
    """Run forward pass into workspace buffers; return the d_fc_out pointer."""
    geom = workspace.geom
    n = x.shape[0]
    if n > workspace.batch_size:
        raise ValueError(f'EvalWorkspace batch_size={workspace.batch_size} < batch {n}')

    upload_to(workspace.d_x, x)

    for i, s in enumerate(geom.conv_stages):
        prev = workspace.d_x if i == 0 else _stage_output_ptr(i - 1, workspace)
        conv_forward_into(prev, device_weights.conv_weights[i],
                          workspace.d_col[i], workspace.d_conv_raw[i],
                          n, s.in_c, s.h_in, s.w_in, s.out_c)
        if s.pool:
            maxpool_forward_into(workspace.d_conv_raw[i], workspace.d_pool[i],
                                 workspace.d_max_idx[i], n, s.out_c, s.h_out, s.w_out)
            cnhw_to_nchw_into(workspace.d_pool[i], workspace.d_pool_nchw[i],
                               n, s.out_c, s.ph, s.pw)
        else:
            cnhw_to_nchw_into(workspace.d_conv_raw[i], workspace.d_conv_nchw[i],
                               n, s.out_c, s.h_out, s.w_out)
            if s.layer_norm:
                ln_idx = geom.ln_param_idx(i)
                lib.layer_norm_forward(
                    workspace.d_ln_out[i],
                    workspace.d_conv_nchw[i],
                    device_weights.ln_gamma[ln_idx],
                    device_weights.ln_beta[ln_idx],
                    n, s.out_c, s.h_out, s.w_out,
                    c_float(1e-5),
                )
            elif s.batch_norm:
                bn_idx = geom.bn_param_idx(i)
                lib.bn_eval_forward(
                    workspace.d_bn_out[i],
                    workspace.d_conv_nchw[i],
                    device_weights.bn_running_mean[bn_idx],
                    device_weights.bn_running_var[bn_idx],
                    device_weights.bn_gamma[bn_idx],
                    device_weights.bn_beta[bn_idx],
                    n, s.out_c, s.h_out, s.w_out,
                    c_float(1e-5),
                )

    fc_in = _stage_output_ptr(geom.n_conv - 1, workspace)
    lib.dense_forward(fc_in, device_weights.fc_w, device_weights.fc_b,
                      workspace.d_fc_out, n, geom.fc_in, geom.fc_out)
    return workspace.d_fc_out


def count_correct_batch_with_workspace(
    x: np.ndarray,
    y: np.ndarray,
    device_weights,
    workspace: EvalWorkspace,
) -> int:
    n = x.shape[0]
    upload_int_to(workspace.d_y, y)
    _forward_logits_into(x, device_weights, workspace)
    zero_bytes(workspace.d_correct, 4)
    lib.count_correct(workspace.d_fc_out, workspace.d_y, workspace.d_correct,
                      n, workspace.geom.fc_out)
    return download_int_scalar(workspace.d_correct)


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    device_weights,
    batch_size: int = BATCH,
    max_batches: int | None = EVAL_MAX_BATCHES,
    workspace: EvalWorkspace | None = None,
) -> float:
    correct = 0
    total = 0
    nbatches = (x.shape[0] + batch_size - 1) // batch_size
    if max_batches is not None:
        nbatches = min(nbatches, max_batches)
    if nbatches <= 0:
        return 0.0
    owns_workspace = workspace is None
    if owns_workspace:
        workspace = EvalWorkspace(batch_size)
    try:
        for i in range(nbatches):
            idx_s = i * batch_size
            idx_e = min(idx_s + batch_size, x.shape[0])
            if idx_s >= idx_e:
                break
            correct += count_correct_batch_with_workspace(
                x[idx_s:idx_e], y[idx_s:idx_e], device_weights, workspace
            )
            total += idx_e - idx_s
    finally:
        if owns_workspace:
            workspace.free()
    return correct / max(total, 1) * 100
