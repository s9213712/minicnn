"""IR node definitions for cuda_native sequential graphs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorSpec:
    """Describes a named tensor with shape and layout metadata."""
    name: str
    shape: tuple[int, ...]
    dtype: str = 'float32'
    layout: str = 'NCHW'

    def numel(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result

    def nbytes(self) -> int:
        _dtype_bytes = {'float32': 4, 'float16': 2, 'int32': 4, 'int64': 8}
        return self.numel() * _dtype_bytes.get(self.dtype, 4)


@dataclass
class Node:
    """A single operation node in the cuda_native graph IR."""
    name: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    input_specs: list[TensorSpec] = field(default_factory=list)
    output_specs: list[TensorSpec] = field(default_factory=list)
    trainable_state: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __repr__(self) -> str:
        in_shapes = [str(s.shape) for s in self.input_specs]
        out_shapes = [str(s.shape) for s in self.output_specs]
        return (
            f'Node({self.op_type!r}, name={self.name!r}, '
            f'in={in_shapes}, out={out_shapes})'
        )
