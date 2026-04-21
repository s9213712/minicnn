# Examples

This folder contains small extension examples for the configuration-driven path.

- `custom_block.py`: custom activation and custom block classes usable from YAML via dotted import paths.
- `mnist_ctypes/`: self-contained MNIST ctypes examples for the handcrafted CUDA shared library.

Most training examples should start from `templates/` rather than editing Python
directly. The legacy CUDA and Torch baseline trainers share common loop helpers
under `src/minicnn/training/loop.py`, and best checkpoints are written under
`artifacts/models/`.

Current config behavior:
- `train.init_seed` controls torch/flex model initialization.
- CLI overrides may address layer-list entries, for example `model.layers.1.out_features=7`.
- String booleans such as `"false"` are parsed strictly for data, augmentation, AMP, and optimizer helper flags.
