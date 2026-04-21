"""Demonstration of StepLR and CosineAnnealingLR schedulers."""
import numpy as np
from minicnn.nn.tensor import Parameter
from minicnn.optim.sgd import SGD
from minicnn.schedulers.cosine import CosineAnnealingLR
from minicnn.schedulers.step import StepLR


def demo_step():
    p = Parameter(np.ones(4, dtype=np.float32))
    opt = SGD([p], lr=0.1)
    sched = StepLR(opt, step_size=5, gamma=0.5)

    print("StepLR (step_size=5, gamma=0.5, initial_lr=0.1):")
    lrs = []
    for epoch in range(13):
        lrs.append(opt.lr)
        sched.step()
    for epoch, lr in enumerate(lrs):
        print(f"  epoch {epoch:2d}: lr={lr:.6f}")


def demo_cosine():
    p = Parameter(np.ones(4, dtype=np.float32))
    opt = SGD([p], lr=0.1)
    T_max = 10
    sched = CosineAnnealingLR(opt, T_max=T_max, lr_min=0.001)

    print(f"\nCosineAnnealingLR (T_max={T_max}, lr_min=0.001, initial_lr=0.1):")
    lrs = []
    for epoch in range(11):
        lrs.append(opt.lr)
        sched.step()
    for epoch, lr in enumerate(lrs):
        print(f"  epoch {epoch:2d}: lr={lr:.6f}")


def main():
    print("=== LR Scheduler Demo ===\n")
    demo_step()
    demo_cosine()
    print("\nAll schedulers ran successfully.")


if __name__ == "__main__":
    main()
