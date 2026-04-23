from __future__ import annotations

import argparse

from minicnn.inference import predict_image, resolve_checkpoint_path
from minicnn._cli_config import _resolve_cli_config_path
from minicnn.unified.config import load_unified_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run single-image inference for a MiniCNN torch/flex checkpoint.'
    )
    parser.add_argument('--config', default='configs/dual_backend_cnn.yaml', help='Path to model config YAML.')
    parser.add_argument('--checkpoint', help='Path to .pt/.pth checkpoint.')
    parser.add_argument('--summary', help='Optional summary.json used to resolve best_model_path.')
    parser.add_argument('--image', required=True, help='Path to a real image file.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('overrides', nargs='*', help='Overrides like dataset.data_root=/path/to/data')
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_unified_config(_resolve_cli_config_path(args.config), args.overrides)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
    )
    result = predict_image(
        cfg,
        checkpoint_path=checkpoint_path,
        image_path=args.image,
        device_name=args.device,
        topk=args.topk,
    )
    prep = result['preprocessing']
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"image: {result['image_path']}")
    print(f"device: {result['device']}")
    print(
        f"preprocess: original={tuple(prep['original_size'])} -> "
        f"target={tuple(prep['target_size'])}, channels={prep['channels']}, dataset={prep['dataset_type']}"
    )
    print('predictions:')
    for entry in result['predictions']:
        print(
            f"  {entry['rank']:>2d}. {entry['label']:<16} "
            f"(index={entry['index']}) prob={entry['probability']:.4f}"
        )


if __name__ == '__main__':
    main()
