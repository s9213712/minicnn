from __future__ import annotations

from minicnn.autograd.context import Context


class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        return cls.forward(ctx, *args, **kwargs)

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):  # pragma: no cover - interface
        raise NotImplementedError
