from __future__ import annotations

from minicnn.autograd.context import Context
from minicnn.nn.tensor import Tensor, _requires_grad


class Function:
    """Base class for custom differentiable operations.

    Subclass this and implement ``forward`` and ``backward``.  Call via
    ``MyOp.apply(*inputs)`` — the backward hook is wired automatically.

    Example::

        class Square(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return Tensor(x.data ** 2, requires_grad=x.requires_grad)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * 2.0 * x.data
    """

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        tensor_inputs = [a for a in args if isinstance(a, Tensor)]
        result = cls.forward(ctx, *args, **kwargs)
        if isinstance(result, Tensor) and _requires_grad(*tensor_inputs):
            result.requires_grad = True
            result._prev = set(tensor_inputs)
            result._op = cls.__name__

            def _backward():
                if result.grad is None:
                    return
                grads = cls.backward(ctx, result.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for tensor, grad in zip(tensor_inputs, grads):
                    if grad is not None and tensor.requires_grad:
                        import numpy as np
                        tensor._add_grad(np.asarray(grad, dtype=np.float32))

            result._backward = _backward
        return result

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):  # pragma: no cover
        raise NotImplementedError
