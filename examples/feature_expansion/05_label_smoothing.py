"""Demonstration of cross_entropy with and without label_smoothing."""
import numpy as np
from minicnn.nn.tensor import Tensor
from minicnn.nn.tensor import cross_entropy

SCENARIOS = [
    ("perfect prediction", np.array([[10.0, -5.0, -5.0]], dtype=np.float32), np.array([0])),
    ("uniform prediction", np.array([[0.0,   0.0,  0.0]], dtype=np.float32), np.array([0])),
    ("wrong prediction",   np.array([[-5.0, 10.0, -5.0]], dtype=np.float32), np.array([0])),
]


def main():
    print("=== Label Smoothing Demo ===\n")
    print(f"{'Scenario':<22}  {'loss (smooth=0)':>16}  {'loss (smooth=0.1)':>18}")
    print("-" * 60)

    for desc, logits_data, targets in SCENARIOS:
        x0 = Tensor(logits_data.copy(), requires_grad=True)
        loss0 = cross_entropy(x0, targets, label_smoothing=0.0)

        x1 = Tensor(logits_data.copy(), requires_grad=True)
        loss1 = cross_entropy(x1, targets, label_smoothing=0.1)

        print(f"{desc:<22}  {float(loss0.data):>16.4f}  {float(loss1.data):>18.4f}")

    print()
    # Show that backward produces valid gradients
    logits_data = np.array([[2.0, 1.0, 0.5]], dtype=np.float32)
    x = Tensor(logits_data.copy(), requires_grad=True)
    loss = cross_entropy(x, np.array([0]), label_smoothing=0.1)
    loss.backward()
    print(f"Gradient with smoothing=0.1 on [[2.0, 1.0, 0.5]] target=0:")
    print(f"  grad = {x.grad.tolist()}")
    assert x.grad is not None
    assert x.grad.shape == logits_data.shape
    print("\nBackward pass produced valid gradients.")
    print("Label smoothing demo ran successfully.")


if __name__ == "__main__":
    main()
