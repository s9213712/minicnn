"""Demonstration of SGD, Adam, AdamW, RMSprop on a toy 2-parameter problem."""
import numpy as np
from minicnn.nn.tensor import Parameter
from minicnn.optim.adam import Adam
from minicnn.optim.adamw import AdamW
from minicnn.optim.rmsprop import RMSprop
from minicnn.optim.sgd import SGD

INIT = np.array([2.0, -1.0], dtype=np.float32)
GRAD = np.array([0.1, -0.1], dtype=np.float32)
STEPS = 5


def run(name, opt_cls, **kwargs):
    p = Parameter(INIT.copy())
    opt = opt_cls([p], **kwargs)
    for _ in range(STEPS):
        p.grad = GRAD.copy()
        opt.step()
    print(f"{name:<12}  params after {STEPS} steps: [{p.data[0]:.6f}, {p.data[1]:.6f}]")


def main():
    print("=== Optimizer Demo ===")
    print(f"Initial params: {INIT.tolist()}, constant grad: {GRAD.tolist()}")
    print()

    run("SGD",     SGD,     lr=0.1)
    run("Adam",    Adam,    lr=0.001)
    run("AdamW",   AdamW,   lr=0.001, weight_decay=0.01)
    run("RMSprop", RMSprop, lr=0.01)

    print()
    print("All optimizers ran successfully.")


if __name__ == "__main__":
    main()
