"""Forward pass and evaluation helpers for the CIFAR-10 CUDA CNN."""

import numpy as np

from minicnn.core.cuda_backend import (
    download_int_scalar,
    lib,
    upload_int,
    zero_bytes,
)
from minicnn.config.settings import (
    BATCH,
    C1_IN,
    C1_OUT,
    C2_IN,
    C2_OUT,
    C3_IN,
    C3_OUT,
    C4_IN,
    C4_OUT,
    EVAL_MAX_BATCHES,
    FC_IN,
    H,
    H1,
    H2,
    H3,
    H4,
    KH,
    KW,
    P1H,
    P1W,
    P2H,
    P2W,
    W,
    W1,
    W2,
    W3,
    W4,
)
from minicnn.training.cuda_ops import cnhw_to_nchw_into, conv_forward_into, malloc_floats, malloc_ints, maxpool_forward_into, upload_int_to, upload_to


def _free_ptrs(ptrs):
    for ptr in reversed(ptrs):
        if ptr:
            lib.gpu_free(ptr)


class EvalWorkspace:
    def __init__(self, batch_size: int):
        self.batch_size = int(batch_size)
        n = self.batch_size
        self.ptrs = []
        self.d_x = self._track(malloc_floats(n * C1_IN * H * W))
        self.d_col1 = self._track(malloc_floats(C1_IN * KH * KW * n * H1 * W1))
        self.d_conv1_raw = self._track(malloc_floats(C1_OUT * n * H1 * W1))
        self.d_conv1_nchw = self._track(malloc_floats(n * C1_OUT * H1 * W1))
        self.d_col2 = self._track(malloc_floats(C2_IN * KH * KW * n * H2 * W2))
        self.d_conv2_raw = self._track(malloc_floats(C2_OUT * n * H2 * W2))
        self.d_pool1 = self._track(malloc_floats(C2_OUT * n * P1H * P1W))
        self.d_max_idx1 = self._track(malloc_ints(C2_OUT * n * P1H * P1W))
        self.d_pool1_nchw = self._track(malloc_floats(n * C2_OUT * P1H * P1W))
        self.d_col3 = self._track(malloc_floats(C3_IN * KH * KW * n * H3 * W3))
        self.d_conv3_raw = self._track(malloc_floats(C3_OUT * n * H3 * W3))
        self.d_conv3_nchw = self._track(malloc_floats(n * C3_OUT * H3 * W3))
        self.d_col4 = self._track(malloc_floats(C4_IN * KH * KW * n * H4 * W4))
        self.d_conv4_raw = self._track(malloc_floats(C4_OUT * n * H4 * W4))
        self.d_pool2 = self._track(malloc_floats(C4_OUT * n * P2H * P2W))
        self.d_max_idx2 = self._track(malloc_ints(C4_OUT * n * P2H * P2W))
        self.d_pool2_nchw = self._track(malloc_floats(n * C4_OUT * P2H * P2W))
        self.d_fc_out = self._track(malloc_floats(n * 10))
        self.d_y = self._track(malloc_ints(n))
        self.d_correct = self._track(malloc_ints(1))

    def _track(self, ptr):
        self.ptrs.append(ptr)
        return ptr

    def free(self) -> None:
        _free_ptrs(self.ptrs)
        self.ptrs = []


def _forward_logits_ptr(x, device_weights, workspace: EvalWorkspace | None = None, ptrs=None):
    d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = device_weights
    n = x.shape[0]
    if workspace is None:
        from minicnn.core.cuda_backend import cnhw_to_nchw_alloc, conv_forward, maxpool_forward, upload

        ptrs = [] if ptrs is None else ptrs
        d_x = upload(x)
        ptrs.append(d_x)

        d_col1, d_conv1_raw, _, _ = conv_forward(d_x, d_w_conv1, n, C1_IN, H, W, C1_OUT)
        ptrs.extend([d_col1, d_conv1_raw])
        d_conv1_nchw = cnhw_to_nchw_alloc(d_conv1_raw, n, C1_OUT, H1, W1)
        ptrs.append(d_conv1_nchw)

        d_col2, d_conv2_raw, _, _ = conv_forward(d_conv1_nchw, d_w_conv2, n, C2_IN, H1, W1, C2_OUT)
        ptrs.extend([d_col2, d_conv2_raw])
        d_pool1, d_max_idx1, _, _ = maxpool_forward(d_conv2_raw, n, C2_OUT, H2, W2)
        ptrs.extend([d_pool1, d_max_idx1])
        d_pool1_nchw = cnhw_to_nchw_alloc(d_pool1, n, C2_OUT, P1H, P1W)
        ptrs.append(d_pool1_nchw)

        d_col3, d_conv3_raw, _, _ = conv_forward(d_pool1_nchw, d_w_conv3, n, C3_IN, P1H, P1W, C3_OUT)
        ptrs.extend([d_col3, d_conv3_raw])
        d_conv3_nchw = cnhw_to_nchw_alloc(d_conv3_raw, n, C3_OUT, H3, W3)
        ptrs.append(d_conv3_nchw)

        d_col4, d_conv4_raw, _, _ = conv_forward(d_conv3_nchw, d_w_conv4, n, C4_IN, H3, W3, C4_OUT)
        ptrs.extend([d_col4, d_conv4_raw])
        d_pool2, d_max_idx2, _, _ = maxpool_forward(d_conv4_raw, n, C4_OUT, H4, W4)
        ptrs.extend([d_pool2, d_max_idx2])
        d_pool2_nchw = cnhw_to_nchw_alloc(d_pool2, n, C4_OUT, P2H, P2W)
        ptrs.append(d_pool2_nchw)

        d_fc_out = lib.gpu_malloc(n * 10 * 4)
        ptrs.append(d_fc_out)
        lib.dense_forward(d_pool2_nchw, d_fc_w, d_fc_b, d_fc_out, n, FC_IN, 10)
        return d_fc_out

    if n > workspace.batch_size:
        raise ValueError(f'EvalWorkspace batch_size={workspace.batch_size} cannot hold batch of {n}')
    upload_to(workspace.d_x, x)
    conv_forward_into(workspace.d_x, d_w_conv1, workspace.d_col1, workspace.d_conv1_raw, n, C1_IN, H, W, C1_OUT)
    cnhw_to_nchw_into(workspace.d_conv1_raw, workspace.d_conv1_nchw, n, C1_OUT, H1, W1)
    conv_forward_into(workspace.d_conv1_nchw, d_w_conv2, workspace.d_col2, workspace.d_conv2_raw, n, C2_IN, H1, W1, C2_OUT)
    maxpool_forward_into(workspace.d_conv2_raw, workspace.d_pool1, workspace.d_max_idx1, n, C2_OUT, H2, W2)
    cnhw_to_nchw_into(workspace.d_pool1, workspace.d_pool1_nchw, n, C2_OUT, P1H, P1W)
    conv_forward_into(workspace.d_pool1_nchw, d_w_conv3, workspace.d_col3, workspace.d_conv3_raw, n, C3_IN, P1H, P1W, C3_OUT)
    cnhw_to_nchw_into(workspace.d_conv3_raw, workspace.d_conv3_nchw, n, C3_OUT, H3, W3)
    conv_forward_into(workspace.d_conv3_nchw, d_w_conv4, workspace.d_col4, workspace.d_conv4_raw, n, C4_IN, H3, W3, C4_OUT)
    maxpool_forward_into(workspace.d_conv4_raw, workspace.d_pool2, workspace.d_max_idx2, n, C4_OUT, H4, W4)
    cnhw_to_nchw_into(workspace.d_pool2, workspace.d_pool2_nchw, n, C4_OUT, P2H, P2W)
    lib.dense_forward(workspace.d_pool2_nchw, d_fc_w, d_fc_b, workspace.d_fc_out, n, FC_IN, 10)
    return workspace.d_fc_out


def forward_batch(x, device_weights):
    n = x.shape[0]
    ptrs = []
    try:
        d_fc_out = _forward_logits_ptr(x, device_weights, ptrs=ptrs)
        h_out = np.zeros((n, 10), dtype=np.float32)
        lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, n * 10 * 4)
        return h_out
    finally:
        _free_ptrs(ptrs)


def count_correct_batch(x, y, device_weights):
    n = x.shape[0]
    ptrs = []
    try:
        d_y = upload_int(y)
        d_correct = lib.gpu_malloc(4)
        ptrs.extend([d_y, d_correct])
        d_fc_out = _forward_logits_ptr(x, device_weights, ptrs=ptrs)
        zero_bytes(d_correct, 4)
        lib.count_correct(d_fc_out, d_y, d_correct, n, 10)
        return download_int_scalar(d_correct)
    finally:
        _free_ptrs(ptrs)


def count_correct_batch_with_workspace(x, y, device_weights, workspace: EvalWorkspace):
    n = x.shape[0]
    upload_int_to(workspace.d_y, y)
    d_fc_out = _forward_logits_ptr(x, device_weights, workspace=workspace)
    zero_bytes(workspace.d_correct, 4)
    lib.count_correct(d_fc_out, workspace.d_y, workspace.d_correct, n, 10)
    return download_int_scalar(workspace.d_correct)


def evaluate(x, y, device_weights, batch_size=BATCH, max_batches=EVAL_MAX_BATCHES):
    correct = 0
    total = 0
    nbatches = (x.shape[0] + batch_size - 1) // batch_size
    if max_batches is not None:
        nbatches = min(nbatches, max_batches)
    if nbatches <= 0:
        return 0.0
    workspace = EvalWorkspace(batch_size)
    try:
        for i in range(nbatches):
            idx_s = i * batch_size
            idx_e = min(idx_s + batch_size, x.shape[0])
            if idx_s >= idx_e:
                break
            correct += count_correct_batch_with_workspace(x[idx_s:idx_e], y[idx_s:idx_e], device_weights, workspace)
            total += idx_e - idx_s
    finally:
        workspace.free()
    if total == 0:
        return 0.0
    return correct / total * 100
