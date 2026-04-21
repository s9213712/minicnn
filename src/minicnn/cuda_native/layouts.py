"""Layout metadata and validation for cuda_native tensors.

Defines canonical layout constants, per-op layout expectations, and helpers
for validating that tensors are in the expected format before execution.

Current cuda_native supports NCHW (4D image tensors) and NC (2D feature
tensors) only. Other layouts are defined as named constants for future use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

#: 4D image tensor: (batch, channels, height, width) — default for conv ops
NCHW = 'NCHW'

#: 4D image tensor: (batch, height, width, channels) — not yet supported
NHWC = 'NHWC'

#: 2D feature tensor: (batch, features) — used after Flatten and in Linear
NC   = 'NC'

#: 1D bias / parameter vector: (features,)
C    = 'C'

#: Scalar or undefined layout
SCALAR = 'SCALAR'

#: All layouts recognized by this module
KNOWN_LAYOUTS: frozenset[str] = frozenset({NCHW, NHWC, NC, C, SCALAR})

#: Layouts currently supported for activation tensors
SUPPORTED_ACTIVATION_LAYOUTS: frozenset[str] = frozenset({NCHW, NC})


# ---------------------------------------------------------------------------
# Per-op layout contract
# ---------------------------------------------------------------------------

#: Maps op_type -> (expected_input_layout, produced_output_layout).
#: None means the layout passes through unchanged.
OP_LAYOUT_CONTRACT: dict[str, tuple[str | None, str | None]] = {
    'Conv2d':     (NCHW,  NCHW),
    'ReLU':       (None,  None),   # passthrough
    'LeakyReLU':  (None,  None),   # passthrough
    'MaxPool2d':  (NCHW,  NCHW),
    'AvgPool2d':  (NCHW,  NCHW),
    'Flatten':    (NCHW,  NC),
    'Linear':     (NC,    NC),
}


def expected_input_layout(op_type: str) -> str | None:
    """Return the expected input activation layout for *op_type*, or None if any layout is accepted."""
    contract = OP_LAYOUT_CONTRACT.get(op_type)
    return contract[0] if contract else None


def expected_output_layout(op_type: str) -> str | None:
    """Return the output layout produced by *op_type*, or None if it mirrors the input."""
    contract = OP_LAYOUT_CONTRACT.get(op_type)
    return contract[1] if contract else None


# ---------------------------------------------------------------------------
# LayoutSpec: attaches layout to a shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutSpec:
    """Pairs a tensor shape with its memory layout tag.

    Provides helpers for rank checking and layout compatibility validation.
    """
    shape: tuple[int, ...]
    layout: str = NCHW

    @property
    def rank(self) -> int:
        return len(self.shape)

    def is_image(self) -> bool:
        """True if this is a 4D NCHW image tensor."""
        return self.rank == 4 and self.layout == NCHW

    def is_flat(self) -> bool:
        """True if this is a 2D NC feature tensor."""
        return self.rank == 2 and self.layout == NC

    def validate(self) -> list[str]:
        """Return a list of layout errors (empty = valid)."""
        errors: list[str] = []
        if self.layout not in KNOWN_LAYOUTS:
            errors.append(f'Unknown layout {self.layout!r}')
        if self.layout == NCHW and self.rank != 4:
            errors.append(f'NCHW layout requires rank 4, got rank {self.rank}')
        if self.layout == NC and self.rank != 2:
            errors.append(f'NC layout requires rank 2, got rank {self.rank}')
        if self.layout == C and self.rank != 1:
            errors.append(f'C layout requires rank 1, got rank {self.rank}')
        return errors


# ---------------------------------------------------------------------------
# Layout inference from shape
# ---------------------------------------------------------------------------

def infer_layout(shape: tuple[int, ...]) -> str:
    """Heuristically infer a layout tag from a tensor shape.

    Rules:
        rank 4 → NCHW
        rank 2 → NC
        rank 1 → C
        other  → SCALAR
    """
    r = len(shape)
    if r == 4:
        return NCHW
    if r == 2:
        return NC
    if r == 1:
        return C
    return SCALAR


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_op_layout(op_type: str, input_layout: str, node_name: str) -> list[str]:
    """Check that *input_layout* satisfies the contract for *op_type*.

    Returns a list of error strings (empty = OK).
    """
    expected = expected_input_layout(op_type)
    if expected is None:
        return []   # passthrough: any layout accepted
    if input_layout != expected:
        return [
            f'Node {node_name} ({op_type}): expected input layout {expected!r}, '
            f'got {input_layout!r}'
        ]
    return []


def validate_graph_layouts(graph) -> list[str]:
    """Walk a NativeGraph and return all layout violations found.

    Infers layout from tensor shape rather than trusting TensorSpec.layout,
    because TensorSpec defaults all tensors to 'NCHW' regardless of rank.
    A rank-2 tensor after Flatten is correctly inferred as NC.
    """
    errors: list[str] = []
    for node in graph.topological_order():
        for spec in node.input_specs:
            layout = infer_layout(spec.shape)
            errors.extend(validate_op_layout(node.op_type, layout, node.name))
    return errors
