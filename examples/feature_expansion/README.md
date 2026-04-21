# Feature Expansion Examples

Self-contained scripts demonstrating features added in phases 0–12.

| File | Description | Requires PyTorch | Run |
|---|---|---|---|
| `01_activations.py` | ReLU, LeakyReLU, SiLU, Tanh, Sigmoid forward+backward | No | `python examples/feature_expansion/01_activations.py` |
| `02_optimizers.py` | SGD, Adam, AdamW, RMSprop on a toy 2-param problem | No | `python examples/feature_expansion/02_optimizers.py` |
| `03_schedulers.py` | StepLR and CosineAnnealingLR lr schedules | No | `python examples/feature_expansion/03_schedulers.py` |
| `04_initialization.py` | All init strategies via `get_initializer()` | No | `python examples/feature_expansion/04_initialization.py` |
| `05_label_smoothing.py` | `cross_entropy` with and without `label_smoothing` | No | `python examples/feature_expansion/05_label_smoothing.py` |
| `06_train_autograd_enhanced.py` | End-to-end: AdamW + SiLU + AvgPool2d + cosine scheduler | No | `python examples/feature_expansion/06_train_autograd_enhanced.py` |
| `07_flex_presets.py` | Block presets `conv_bn_relu`, `conv_bn_silu` in flex builder | Yes | `python examples/feature_expansion/07_flex_presets.py` |
| `08_augmentation.py` | Augmentation config in `create_dataloaders()` | Yes | `python examples/feature_expansion/08_augmentation.py` |

Scripts 01–06 require only NumPy. Scripts 07–08 skip gracefully if PyTorch is absent.
