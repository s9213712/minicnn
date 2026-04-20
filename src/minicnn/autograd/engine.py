from __future__ import annotations

from minicnn.nn.tensor import Tensor


def backward(tensor: Tensor, grad=None) -> None:
    """Trigger reverse-mode autodiff on ``tensor``.

    This is a thin convenience wrapper around ``Tensor.backward()``.
    Prefer calling ``tensor.backward()`` directly.
    """
    tensor.backward(grad)
