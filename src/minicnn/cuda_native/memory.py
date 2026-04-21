"""Memory allocation helpers for cuda_native.

Provides a BufferAllocator that pre-allocates numpy arrays according to
an ExecutionPlan's BufferPlan, and an arena-style pool that the executor
can optionally use instead of creating new arrays on every call.

This module is intentionally simple — it does not manage CUDA memory or
C-level arenas. It is a numpy-level abstraction designed to:
  1. Make memory allocation inspectable and testable independently
  2. Provide a seam for future C-level or CUDA allocator integration
  3. Enable buffer reuse across calls without the executor knowing the details
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# BufferAllocator: materialises a BufferPlan into numpy arrays
# ---------------------------------------------------------------------------

@dataclass
class BufferAllocator:
    """Allocates numpy arrays for every buffer described in a BufferPlan.

    Usage::

        plan = make_naive_plan(graph)
        allocator = BufferAllocator.from_plan(plan)
        pool = allocator.allocate()   # dict[buf_name -> np.ndarray]
    """
    buffer_specs: dict[str, tuple[int, ...]]   # buf_name -> shape
    dtype: np.dtype = field(default_factory=lambda: np.dtype('float32'))

    @classmethod
    def from_plan(cls, plan, dtype: str = 'float32') -> 'BufferAllocator':
        """Build an allocator from an ExecutionPlan.

        Reconstructs shapes from byte sizes assuming a uniform dtype.
        For a mixed-dtype plan, use from_plan_with_specs() instead.
        """
        _bytes_per_elem = {'float32': 4, 'float16': 2, 'int32': 4, 'int64': 8}
        bpe = _bytes_per_elem.get(dtype, 4)
        specs: dict[str, tuple[int, ...]] = {}
        for buf_name, nbytes in plan.buffer_plan.buffer_nbytes.items():
            numel = nbytes // bpe
            specs[buf_name] = (numel,)   # flat shape; executor fills real shape
        return cls(buffer_specs=specs, dtype=np.dtype(dtype))

    @classmethod
    def from_tensor_specs(cls, plan, graph, dtype: str = 'float32') -> 'BufferAllocator':
        """Build an allocator that allocates buffers with the correct N-D shape.

        Looks up each buffer's tensor via the plan's tensor_to_buffer mapping,
        then reads the shape from the graph's node specs.
        """
        tensor_shapes: dict[str, tuple[int, ...]] = {}
        if graph.input_spec:
            tensor_shapes[graph.input_spec.name] = graph.input_spec.shape
        for node in graph.topological_order():
            for spec in node.output_specs:
                tensor_shapes[spec.name] = spec.shape

        buf_to_tensor = {v: k for k, v in plan.buffer_plan.tensor_to_buffer.items()}
        specs: dict[str, tuple[int, ...]] = {}
        for buf_name in plan.buffer_plan.buffer_nbytes:
            tensor_name = buf_to_tensor.get(buf_name)
            if tensor_name and tensor_name in tensor_shapes:
                specs[buf_name] = tensor_shapes[tensor_name]
            else:
                nbytes = plan.buffer_plan.buffer_nbytes[buf_name]
                bpe = {'float32': 4, 'float16': 2, 'int32': 4, 'int64': 8}.get(dtype, 4)
                specs[buf_name] = (nbytes // bpe,)
        return cls(buffer_specs=specs, dtype=np.dtype(dtype))

    def allocate(self) -> dict[str, np.ndarray]:
        """Allocate and return a fresh pool of zero-filled numpy arrays."""
        return {
            name: np.zeros(shape, dtype=self.dtype)
            for name, shape in self.buffer_specs.items()
        }

    @property
    def total_bytes(self) -> int:
        """Total bytes across all buffers (based on dtype element size)."""
        bpe = self.dtype.itemsize
        return sum(int(np.prod(s)) * bpe for s in self.buffer_specs.values())

    @property
    def num_buffers(self) -> int:
        return len(self.buffer_specs)

    def summary(self) -> dict[str, Any]:
        return {
            'num_buffers': self.num_buffers,
            'total_bytes': self.total_bytes,
            'total_kb': round(self.total_bytes / 1024, 2),
            'dtype': str(self.dtype),
            'buffers': {name: list(shape) for name, shape in self.buffer_specs.items()},
        }


# ---------------------------------------------------------------------------
# BufferPool: a pre-allocated pool reusable across inference calls
# ---------------------------------------------------------------------------

class BufferPool:
    """A pre-allocated pool of numpy arrays that can be reused across calls.

    Avoids repeated allocation overhead when running many inference calls
    with the same graph shape.

    Usage::

        pool = BufferPool.build(plan, graph)
        ctx = pool.make_ctx(feeds={'input': x}, params=params)
        # pass ctx to ForwardExecutor.run() ... (future integration)
        pool.reset()   # zero-fill all buffers for next use
    """

    def __init__(self, allocator: BufferAllocator, plan):
        self._allocator = allocator
        self._plan = plan
        self._pool: dict[str, np.ndarray] = allocator.allocate()

    @classmethod
    def build(cls, plan, graph, dtype: str = 'float32') -> 'BufferPool':
        """Build a pool from an ExecutionPlan and a NativeGraph."""
        allocator = BufferAllocator.from_tensor_specs(plan, graph, dtype=dtype)
        return cls(allocator, plan)

    def reset(self) -> None:
        """Zero-fill all buffers, ready for the next call."""
        for arr in self._pool.values():
            arr.fill(0.0)

    def make_ctx(
        self,
        feeds: dict[str, np.ndarray],
        params: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Build an execution context dict seeded with feeds and params.

        The pool buffers are included as hints for future buffer-aware executors.
        Standard ForwardExecutor ignores them and allocates its own arrays, but
        a future buffer-aware executor can write directly into pool arrays.
        """
        ctx: dict[str, Any] = {}
        ctx.update(self._pool)    # pre-allocated buffers as starting point
        ctx.update(feeds)         # feeds override pool slots
        if params:
            ctx.update(params)
        return ctx

    @property
    def total_bytes(self) -> int:
        return self._allocator.total_bytes

    @property
    def num_buffers(self) -> int:
        return self._allocator.num_buffers

    def summary(self) -> dict[str, Any]:
        return self._allocator.summary()


# ---------------------------------------------------------------------------
# Convenience: memory_footprint
# ---------------------------------------------------------------------------

def memory_footprint(graph, dtype: str = 'float32') -> dict[str, Any]:
    """Return a dict describing the naive memory footprint of *graph*.

    Does not require building a plan — uses shape information from the graph
    directly. Useful for quick estimates before committing to a full plan.
    """
    from minicnn.cuda_native.planner import make_naive_plan
    plan = make_naive_plan(graph)
    allocator = BufferAllocator.from_tensor_specs(plan, graph, dtype=dtype)
    return allocator.summary()
