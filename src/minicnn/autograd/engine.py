from __future__ import annotations


def backward(tensor, grad=None) -> None:
    tensor.backward(grad)
