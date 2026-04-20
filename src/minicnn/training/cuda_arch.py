"""Centralised architecture geometry for the CUDA CNN backend.

A single YAML `conv_layers` list is the only place that needs to change when
the network architecture changes.  Everything else — workspace buffer sizes,
weight shapes, checkpoint keys, forward/backward loop bounds — is derived from
`CudaNetGeometry` at startup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from minicnn.config.parsing import parse_bool

if TYPE_CHECKING:
    from minicnn.config.schema import ModelConfig


@dataclass(frozen=True)
class ConvStage:
    """Geometry of one conv-[leaky_relu]-[maxpool] stage."""

    idx: int
    in_c: int
    out_c: int
    kh: int
    kw: int
    h_in: int   # spatial height fed into this conv
    w_in: int
    h_out: int  # spatial height after conv, BEFORE optional pool
    w_out: int
    pool: bool  # is there a 2×2 maxpool after the conv?
    ph: int     # spatial height AFTER pool  (= h_out // 2 if pool else h_out)
    pw: int

    @property
    def weight_numel(self) -> int:
        return self.out_c * self.in_c * self.kh * self.kw

    @property
    def col_numel_per_sample(self) -> int:
        """Elements in the im2col output for one sample."""
        return self.in_c * self.kh * self.kw * self.h_out * self.w_out


class CudaNetGeometry:
    """All shape constants for the CUDA training pipeline.

    Replaces the scattered per-layer globals (C1_IN, C1_OUT, H1, W1, P1H …)
    that previously had to be kept in sync across six files.
    """

    def __init__(self, conv_stages: list[ConvStage], fc_in: int, fc_out: int) -> None:
        self.conv_stages = conv_stages
        self.fc_in = fc_in
        self.fc_out = fc_out

    @classmethod
    def from_config(cls, model_cfg: ModelConfig) -> CudaNetGeometry:
        kh, kw = model_cfg.kh, model_cfg.kw
        h, w, c_in = model_cfg.h, model_cfg.w, model_cfg.c_in
        fc_out = model_cfg.fc_out

        stages: list[ConvStage] = []
        cur_h, cur_w, cur_c = h, w, c_in
        for i, spec in enumerate(model_cfg.conv_layers):
            out_c = int(spec['out_c'])
            pool = parse_bool(spec.get('pool', False), label=f'model.conv_layers[{i}].pool')
            h_out = cur_h - kh + 1
            w_out = cur_w - kw + 1
            if h_out <= 0 or w_out <= 0:
                raise ValueError(
                    f"Conv stage {i}: input {cur_h}×{cur_w}, kernel {kh}×{kw} "
                    f"→ invalid output {h_out}×{w_out}"
                )
            if pool and (h_out % 2 != 0 or w_out % 2 != 0):
                raise ValueError(
                    f"Conv stage {i}: pool=True requires even spatial dims, got {h_out}×{w_out}"
                )
            ph = h_out // 2 if pool else h_out
            pw = w_out // 2 if pool else w_out
            stages.append(ConvStage(i, cur_c, out_c, kh, kw, cur_h, cur_w, h_out, w_out, pool, ph, pw))
            cur_c, cur_h, cur_w = out_c, ph, pw

        fc_in = cur_c * cur_h * cur_w
        if fc_in <= 0:
            raise ValueError(f"Computed FC input size {fc_in} ≤ 0")
        return cls(stages, fc_in, fc_out)

    @property
    def n_conv(self) -> int:
        return len(self.conv_stages)

    def stage_output_nchw_dims(self, i: int) -> tuple[int, int, int]:
        """(channels, h, w) of the NCHW output of stage i (after pool if applicable)."""
        s = self.conv_stages[i]
        return s.out_c, s.ph, s.pw
