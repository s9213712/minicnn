"""Demonstration of augmentation config in flex/data.py. Requires PyTorch."""
try:
    import torch
except ImportError:
    print("PyTorch not available; skipping 08_augmentation.py")
    raise SystemExit(0)

from minicnn.flex.data import create_dataloaders


def main():
    print("=== Augmentation DataLoader Demo ===\n")

    dataset_cfg = {
        'type': 'random',
        'num_samples': 64,
        'val_samples': 16,
        'input_shape': [3, 16, 16],
        'num_classes': 4,
    }
    train_cfg = {'batch_size': 8, 'num_workers': 0}
    augmentation_cfg = {
        'random_crop': True,
        'random_crop_padding': 2,
        'horizontal_flip': True,
    }

    train_loader, val_loader = create_dataloaders(
        dataset_cfg, train_cfg, augmentation_cfg=augmentation_cfg
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader:   {len(val_loader)} batches")

    first_batch_x, first_batch_y = next(iter(train_loader))
    print(f"\nFirst train batch shape: x={list(first_batch_x.shape)}, y={list(first_batch_y.shape)}")

    print("\nAugmentation demo ran successfully.")


if __name__ == "__main__":
    main()
