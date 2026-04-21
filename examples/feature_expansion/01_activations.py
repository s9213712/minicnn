"""Demonstration of all autograd activations: ReLU, LeakyReLU, SiLU, Tanh, Sigmoid."""
import numpy as np
from minicnn.nn.layers import LeakyReLU, ReLU, Sigmoid, SiLU, Tanh
from minicnn.nn.tensor import Tensor


def demo(name, module, x_data):
    x = Tensor(x_data.copy(), requires_grad=True)
    y = module(x)
    y.sum().backward()
    print(f"{name:<12}  out_shape={y.shape}  grad_shape={x.grad.shape}"
          f"  out[0]={y.data.flat[0]:.4f}  grad[0]={x.grad.flat[0]:.4f}")


def main():
    x_data = np.array([[-1.0, 0.0, 0.5, 2.0]], dtype=np.float32)

    print("=== Autograd Activations Demo ===")
    print(f"Input: {x_data}")
    print()

    demo("ReLU",      ReLU(),                  x_data)
    demo("LeakyReLU", LeakyReLU(0.01),         x_data)
    demo("SiLU",      SiLU(),                  x_data)
    demo("Tanh",      Tanh(),                  x_data)
    demo("Sigmoid",   Sigmoid(),               x_data)

    print()
    print("All activations ran successfully.")


if __name__ == "__main__":
    main()
