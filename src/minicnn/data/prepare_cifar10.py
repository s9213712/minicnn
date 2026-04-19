from minicnn.data.cifar10 import prepare_cifar10
from minicnn.paths import DATA_ROOT


def main() -> None:
    path = prepare_cifar10(DATA_ROOT, download=True)
    print(f"CIFAR-10 ready at {path}")


if __name__ == "__main__":
    main()
