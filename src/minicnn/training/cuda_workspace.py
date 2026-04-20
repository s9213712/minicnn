"""Reusable GPU workspace for one CUDA CNN training batch.

All buffer sizes are derived from CudaNetGeometry, so changing `conv_layers`
in the YAML is all that is needed to adjust the network depth.
"""
from __future__ import annotations

from minicnn.config.settings import BATCH, get_arch
from minicnn.core.cuda_backend import lib
from minicnn.training.cuda_ops import malloc_floats, malloc_ints


class BatchWorkspace:
    def __init__(self) -> None:
        geom = get_arch()
        N = BATCH
        self.ptrs: list = []
        self._freed = False

        # ---- input ----
        s0 = geom.conv_stages[0]
        self.d_x = self.alloc(s0.in_c * N * s0.h_in * s0.w_in)

        # ---- per-stage forward buffers (indexed by stage) ----
        self.d_col: list       = []   # im2col output
        self.d_conv_raw: list  = []   # conv output CNHW (before LN/BN/activation)
        self.d_ln_out: list    = []   # LayerNorm output NCHW (LN stages only, else None)
        self.d_bn_out: list    = []   # BatchNorm output NCHW (BN stages only, else None)
        self.d_bn_x_hat: list  = []   # saved x_hat for BN backward (BN stages only, else None)
        self.d_bn_mean: list   = []   # saved batch mean per channel (BN stages only, else None)
        self.d_bn_inv_std: list = []  # saved batch inv_std per channel (BN stages only, else None)
        self.d_conv_nchw: list = []   # NCHW version (non-pool stages only, else None)
        self.d_pool: list      = []   # pool output CNHW (pool stages only, else None)
        self.d_max_idx: list   = []   # pool argmax (pool stages only, else None)
        self.d_pool_nchw: list = []   # pool output NCHW (pool stages only, else None)

        for s in geom.conv_stages:
            self.d_col.append(self.alloc(s.in_c * s.kh * s.kw * N * s.h_out * s.w_out))
            self.d_conv_raw.append(self.alloc(s.out_c * N * s.h_out * s.w_out))
            # LN output buffer only needed for non-pool LN stages.
            if s.layer_norm and not s.pool:
                self.d_ln_out.append(self.alloc(s.out_c * N * s.h_out * s.w_out))
            else:
                self.d_ln_out.append(None)
            # BN buffers: output, saved x_hat, mean, inv_std (BN stages only).
            if s.batch_norm:
                spatial = N * s.out_c * s.h_out * s.w_out
                self.d_bn_out.append(self.alloc(spatial))
                self.d_bn_x_hat.append(self.alloc(spatial))
                self.d_bn_mean.append(self.alloc(s.out_c))
                self.d_bn_inv_std.append(self.alloc(s.out_c))
            else:
                self.d_bn_out.append(None)
                self.d_bn_x_hat.append(None)
                self.d_bn_mean.append(None)
                self.d_bn_inv_std.append(None)
            if s.pool:
                self.d_pool.append(self.alloc(s.out_c * N * s.ph * s.pw))
                self.d_max_idx.append(self.alloc_ints(s.out_c * N * s.ph * s.pw))
                self.d_pool_nchw.append(self.alloc(N * s.out_c * s.ph * s.pw))
                self.d_conv_nchw.append(None)
            else:
                self.d_conv_nchw.append(self.alloc(N * s.out_c * s.h_out * s.w_out))
                self.d_pool.append(None)
                self.d_max_idx.append(None)
                self.d_pool_nchw.append(None)

        # ---- FC forward/loss buffers ----
        self.d_fc_out       = self.alloc(N * geom.fc_out)
        self.d_y            = self.alloc_ints(N)
        self.d_probs        = self.alloc(N * geom.fc_out)
        self.d_grad_logits  = self.alloc(N * geom.fc_out)
        self.d_loss_sum     = self.alloc(1)
        self.d_correct      = self.alloc_ints(1)

        # ---- FC backward buffers ----
        self.d_pre_fc_grad_nchw = self.alloc(N * geom.fc_in)   # grad wrt FC input
        self.d_fc_grad_w        = self.alloc(geom.fc_out * geom.fc_in)
        self.d_fc_grad_b        = self.alloc(geom.fc_out)

        # ---- per-stage backward buffers ----
        self.d_w_grad: list            = []  # weight gradients
        self.d_conv_raw_grad: list     = []  # grad wrt conv output (CNHW)
        self.d_ln_input_grad: list     = []  # grad wrt LN input (= conv output NCHW) when LN present
        self.d_bn_input_grad: list     = []  # grad wrt BN input (= d_conv_nchw) when BN present
        self.d_bn_dgamma: list         = []  # accumulated BN gamma gradient (BN stages only)
        self.d_bn_dbeta: list          = []  # accumulated BN beta gradient (BN stages only)
        self.d_pool_grad_cnhw: list    = []  # grad wrt pool output (CNHW, pool stages only)
        self.d_input_nchw_grad: list   = []  # grad wrt stage input (NCHW)

        for s in geom.conv_stages:
            self.d_w_grad.append(self.alloc(s.weight_numel))
            self.d_conv_raw_grad.append(self.alloc(s.out_c * N * s.h_out * s.w_out))
            if s.layer_norm and not s.pool:
                self.d_ln_input_grad.append(self.alloc(s.out_c * N * s.h_out * s.w_out))
            else:
                self.d_ln_input_grad.append(None)
            if s.batch_norm:
                spatial = N * s.out_c * s.h_out * s.w_out
                self.d_bn_input_grad.append(self.alloc(spatial))
                self.d_bn_dgamma.append(self.alloc(s.out_c))
                self.d_bn_dbeta.append(self.alloc(s.out_c))
            else:
                self.d_bn_input_grad.append(None)
                self.d_bn_dgamma.append(None)
                self.d_bn_dbeta.append(None)
            if s.pool:
                self.d_pool_grad_cnhw.append(self.alloc(s.out_c * N * s.ph * s.pw))
            else:
                self.d_pool_grad_cnhw.append(None)
            self.d_input_nchw_grad.append(self.alloc(s.in_c * N * s.h_in * s.w_in))

    def alloc(self, size: int):
        if self._freed:
            raise RuntimeError('Cannot allocate from a freed BatchWorkspace')
        ptr = malloc_floats(size)
        self.ptrs.append(ptr)
        return ptr

    def alloc_ints(self, size: int):
        if self._freed:
            raise RuntimeError('Cannot allocate from a freed BatchWorkspace')
        ptr = malloc_ints(size)
        self.ptrs.append(ptr)
        return ptr

    def free(self) -> None:
        if self._freed:
            return
        for ptr in self.ptrs:
            lib.gpu_free(ptr)
        self.ptrs = []
        self._freed = True

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.free()
        except Exception as exc:
            import warnings
            warnings.warn(
                f"BatchWorkspace.__del__: GPU memory cleanup failed: {exc}",
                ResourceWarning,
                stacklevel=2,
            )
