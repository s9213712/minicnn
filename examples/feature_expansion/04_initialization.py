"""Demonstration of weight initialization strategies via get_initializer()."""
import numpy as np
from minicnn.models.initialization import get_initializer

SHAPE = (64, 32, 3, 3)

STRATEGIES = [
    'kaiming_uniform',
    'kaiming_normal',
    'xavier_uniform',
    'xavier_normal',
    'normal',
    'zeros',
]


def main():
    print("=== Weight Initialization Demo ===")
    print(f"Shape: {SHAPE}\n")
    print(f"{'Strategy':<18}  {'shape':<18}  {'mean':>10}  {'std':>10}")
    print("-" * 62)

    for name in STRATEGIES:
        init_fn = get_initializer(name)
        w = init_fn(SHAPE)
        mean = float(np.mean(w))
        std = float(np.std(w))
        print(f"{name:<18}  {str(w.shape):<18}  {mean:>10.6f}  {std:>10.6f}")

    print("\nAll initializers ran successfully.")


if __name__ == "__main__":
    main()
