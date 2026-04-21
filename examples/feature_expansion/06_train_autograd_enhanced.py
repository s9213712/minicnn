"""End-to-end training with AdamW, SiLU, AvgPool2d, label smoothing, cosine scheduler."""
from minicnn.training.train_autograd import train_autograd_from_config


def main():
    cfg = {
        'dataset': {
            'type': 'random',
            'num_samples': 64,
            'val_samples': 16,
            'input_shape': [1, 8, 8],
            'num_classes': 4,
        },
        'train': {'epochs': 3, 'batch_size': 8},
        'model': {
            'input_shape': [1, 8, 8],
            'layers': [
                {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 3},
                {'type': 'SiLU'},
                {'type': 'AvgPool2d', 'kernel_size': 2},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 4},
            ],
        },
        'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01},
        'loss': {'type': 'CrossEntropyLoss', 'label_smoothing': 0.1},
        'scheduler': {'enabled': True, 'type': 'cosine', 'T_max': 3, 'min_lr': 0.0001},
    }

    print("=== Enhanced Autograd Training Demo ===\n")
    run_dir = train_autograd_from_config(cfg)
    print(f"\nRun directory: {run_dir}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
