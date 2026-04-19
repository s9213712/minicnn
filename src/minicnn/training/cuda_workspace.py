"""Reusable GPU workspace for one legacy CUDA CIFAR-10 batch."""

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
from minicnn.core.cuda_backend import lib
from minicnn.training.cuda_ops import malloc_floats, malloc_ints


class BatchWorkspace:
    def __init__(self):
        self.ptrs = []
        self._freed = False
        self.d_x = self.alloc(C1_IN * BATCH * H * W)
        self.d_col1 = self.alloc(C1_IN * KH * KW * BATCH * H1 * W1)
        self.d_conv1_raw = self.alloc(C1_OUT * BATCH * H1 * W1)
        self.d_conv1_nchw = self.alloc(BATCH * C1_OUT * H1 * W1)
        self.d_col2 = self.alloc(C2_IN * KH * KW * BATCH * H2 * W2)
        self.d_conv2_raw = self.alloc(C2_OUT * BATCH * H2 * W2)
        self.d_pool1 = self.alloc(C2_OUT * BATCH * P1H * P1W)
        self.d_max_idx1 = self.alloc_ints(C2_OUT * BATCH * P1H * P1W)
        self.d_pool1_nchw = self.alloc(BATCH * C2_OUT * P1H * P1W)
        self.d_col3 = self.alloc(C3_IN * KH * KW * BATCH * H3 * W3)
        self.d_conv3_raw = self.alloc(C3_OUT * BATCH * H3 * W3)
        self.d_conv3_nchw = self.alloc(BATCH * C3_OUT * H3 * W3)
        self.d_col4 = self.alloc(C4_IN * KH * KW * BATCH * H4 * W4)
        self.d_conv4_raw = self.alloc(C4_OUT * BATCH * H4 * W4)
        self.d_pool2 = self.alloc(C4_OUT * BATCH * P2H * P2W)
        self.d_max_idx2 = self.alloc_ints(C4_OUT * BATCH * P2H * P2W)
        self.d_pool2_nchw = self.alloc(BATCH * C4_OUT * P2H * P2W)
        self.d_fc_out = self.alloc(BATCH * 10)
        self.d_y = self.alloc_ints(BATCH)
        self.d_probs = self.alloc(BATCH * 10)
        self.d_grad_logits = self.alloc(BATCH * 10)
        self.d_loss_sum = self.alloc(1)
        self.d_correct = self.alloc_ints(1)
        self.d_pool2_grad_nchw = self.alloc(BATCH * FC_IN)
        self.d_fc_grad_w = self.alloc(10 * FC_IN)
        self.d_fc_grad_b = self.alloc(10)
        self.d_pool2_grad = self.alloc(C4_OUT * BATCH * P2H * P2W)
        self.d_conv4_grad_raw = self.alloc(C4_OUT * BATCH * H4 * W4)
        self.d_w_conv4_grad = self.alloc(C4_OUT * C4_IN * KH * KW)
        self.d_conv3_grad = self.alloc(C4_IN * BATCH * H3 * W3)
        self.d_conv3_grad_raw = self.alloc(C3_OUT * BATCH * H3 * W3)
        self.d_w_conv3_grad = self.alloc(C3_OUT * C3_IN * KH * KW)
        self.d_pool1_grad = self.alloc(C3_IN * BATCH * P1H * P1W)
        self.d_pool1_grad_cnhw = self.alloc(C2_OUT * BATCH * P1H * P1W)
        self.d_conv2_grad_raw = self.alloc(C2_OUT * BATCH * H2 * W2)
        self.d_w_conv2_grad = self.alloc(C2_OUT * C2_IN * KH * KW)
        self.d_conv1_grad = self.alloc(C2_IN * BATCH * H1 * W1)
        self.d_conv1_grad_raw = self.alloc(C1_OUT * BATCH * H1 * W1)
        self.d_w_conv1_grad = self.alloc(C1_OUT * C1_IN * KH * KW)
        self.d_x_grad = self.alloc(C1_IN * BATCH * H * W)

    def alloc(self, size):
        if self._freed:
            raise RuntimeError('Cannot allocate from a freed BatchWorkspace')
        ptr = malloc_floats(size)
        self.ptrs.append(ptr)
        return ptr

    def alloc_ints(self, size):
        if self._freed:
            raise RuntimeError('Cannot allocate from a freed BatchWorkspace')
        ptr = malloc_ints(size)
        self.ptrs.append(ptr)
        return ptr

    def free(self):
        if self._freed:
            return
        for ptr in self.ptrs:
            lib.gpu_free(ptr)
        self.ptrs = []
        self._freed = True

    def __del__(self):  # pragma: no cover - defensive cleanup only
        try:
            self.free()
        except Exception:
            pass
