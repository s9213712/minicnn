from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Callable, Iterator
import threading

import numpy as np

_grad_ctx = threading.local()


@contextmanager
def no_grad() -> Iterator[None]:
    prev = getattr(_grad_ctx, 'enabled', True)
    _grad_ctx.enabled = False
    try:
        yield
    finally:
        _grad_ctx.enabled = prev


def is_grad_enabled() -> bool:
    return getattr(_grad_ctx, 'enabled', True)


def _array(data: Any) -> np.ndarray:
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    grad = np.asarray(grad, dtype=np.float32)
    if shape == ():
        return np.asarray(grad.sum(), dtype=np.float32)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _requires_grad(*tensors: 'Tensor') -> bool:
    return is_grad_enabled() and any(t.requires_grad for t in tensors)


@dataclass(eq=False)
class Tensor:
    data: Any
    grad: Any = None
    requires_grad: bool = False
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.data = _array(self.data)
        self.requires_grad = bool(self.requires_grad)
        self._prev: set[Tensor] = set()
        self._backward: Callable[[], None] = lambda: None
        self._op: str = ''

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, grad={self.grad!r}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = None

    def detach(self) -> 'Tensor':
        return Tensor(self.data.copy(), requires_grad=False, name=self.name, metadata=dict(self.metadata))

    def _add_grad(self, grad: Any) -> None:
        if not self.requires_grad:
            return
        grad = np.asarray(grad, dtype=np.float32)
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, grad: Any = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError('grad must be provided for non-scalar Tensor.backward()')
            grad = np.ones_like(self.data, dtype=np.float32)
        else:
            grad = np.asarray(grad, dtype=np.float32)
            if grad.shape != self.data.shape:
                raise ValueError(f'backward grad shape {grad.shape} does not match tensor shape {self.data.shape}')

        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                if node not in visited:
                    visited.add(node)
                    topo.append(node)
            elif node not in visited:
                stack.append((node, True))
                for child in node._prev:
                    stack.append((child, False))

        self.grad = grad
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: Any) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=_requires_grad(self, other))
        out._prev = {self, other}
        out._op = 'add'

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(_unbroadcast(out.grad, self.data.shape))
            other._add_grad(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> 'Tensor':
        return self + other

    def __sub__(self, other: Any) -> 'Tensor':
        return self + (-other)

    def __rsub__(self, other: Any) -> 'Tensor':
        return other + (-self)

    def __neg__(self) -> 'Tensor':
        out = Tensor(-self.data, requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'neg'

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(-out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=_requires_grad(self, other))
        out._prev = {self, other}
        out._op = 'mul'

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(_unbroadcast(out.grad * other.data, self.data.shape))
            other._add_grad(_unbroadcast(out.grad * self.data, other.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> 'Tensor':
        return self * other

    def __truediv__(self, other: Any) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Any) -> 'Tensor':
        return other * (self ** -1.0)

    def __pow__(self, power: float) -> 'Tensor':
        out = Tensor(self.data ** power, requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'pow'

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * power * (self.data ** (power - 1.0)))

        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=_requires_grad(self, other))
        out._prev = {self, other}
        out._op = 'matmul'

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(out.grad @ np.swapaxes(other.data, -1, -2))
            other._add_grad(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'sum'

        def _backward() -> None:
            if out.grad is None:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self._add_grad(np.ones_like(self.data, dtype=np.float32) * grad)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> 'Tensor':
        if axis is None:
            denom = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            denom = 1
            for ax in axes:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def reshape(self, *shape: int | tuple[int, ...]) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'reshape'

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'relu'

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'exp'

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * out_data)

        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data + 1e-10), requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'log'

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad / (self.data + 1e-10))

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out_data = shifted - logsumexp
        out = Tensor(out_data, requires_grad=_requires_grad(self))
        out._prev = {self}
        out._op = 'log_softmax'

        def _backward() -> None:
            if out.grad is None:
                return
            probs = np.exp(out_data)
            grad = out.grad - probs * out.grad.sum(axis=axis, keepdims=True)
            self._add_grad(grad)

        out._backward = _backward
        return out


class Parameter(Tensor):
    def __init__(self, data: Any, name: str | None = None, metadata: dict[str, Any] | None = None):
        super().__init__(data=data, requires_grad=True, name=name, metadata=metadata or {})


def relu(x: Tensor) -> Tensor:
    return x.relu()


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis)


def cross_entropy(logits: Tensor, targets: Any) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64)
    if logits.data.ndim != 2:
        raise ValueError(f'cross_entropy expects logits with shape (N, C), got {logits.data.shape}')
    if targets.shape != (logits.data.shape[0],):
        raise ValueError(f'cross_entropy targets must have shape ({logits.data.shape[0]},), got {targets.shape}')
    n = logits.data.shape[0]
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss_val = -np.log(probs[np.arange(n), targets] + 1e-10).sum() / float(n)
    out = Tensor(loss_val, requires_grad=_requires_grad(logits))
    out._prev = {logits}
    out._op = 'cross_entropy'

    def _backward() -> None:
        if out.grad is None:
            return
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0
        grad *= np.asarray(out.grad, dtype=np.float32) / float(n)
        logits._add_grad(grad)

    out._backward = _backward
    return out
