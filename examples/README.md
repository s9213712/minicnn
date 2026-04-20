# Examples

This folder contains small extension examples for the configuration-driven path.

- `custom_block.py`: custom activation and custom block classes usable from YAML via dotted import paths.

Most training examples should start from `templates/` rather than editing Python
directly. The legacy CUDA and Torch baseline trainers share common loop helpers
under `src/minicnn/training/loop.py`, and best checkpoints are written under
`src/minicnn/training/models/`.
