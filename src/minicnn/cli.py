from __future__ import annotations

import argparse
import json
import time

from minicnn.core.build import build_native, check_native
from minicnn.flex.config import load_flex_config, dump_template
from minicnn.flex.registry import describe_registries
from minicnn.flex.trainer import train_from_config
from minicnn.framework.health import doctor, healthcheck
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT
from minicnn.unified.config import load_unified_config, dump_unified_template
from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED, summarize_legacy_mapping, validate_cuda_legacy_compatibility
from minicnn.unified.trainer import train_unified_from_config


def _load_flex_config_or_exit(config_path: str, overrides: list[str]) -> dict:
    try:
        return load_flex_config(config_path, overrides)
    except FileNotFoundError:
        print(
            f"Config file not found: {config_path}\n"
            "Create a template with:\n"
            "  minicnn config-template > configs/my_config.yaml"
        )
        raise SystemExit(2)


def _load_unified_config_or_exit(config_path: str, overrides: list[str]) -> dict:
    try:
        return load_unified_config(config_path, overrides)
    except FileNotFoundError:
        print(
            f"Config file not found: {config_path}\n"
            "Create a template with:\n"
            "  minicnn dual-config-template > configs/my_config.yaml"
        )
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='minicnn', description='MiniCNN dual-backend CLI (pure handcrafted CUDA + PyTorch)')
    sub = parser.add_subparsers(dest='command', required=True)

    p_build = sub.add_parser('build', help='Build CUDA shared library')
    p_build.add_argument('--no-cublas', action='store_true', help='Disable cuBLAS and use handwritten fallback')
    p_build.add_argument('--check', action='store_true', help='Run symbol/export checks after build')
    p_build.add_argument('--generator', choices=['make', 'ninja'], default='make')
    p_build.add_argument('--legacy-make', action='store_true', help='Use original cpp/Makefile instead of CMake')
    p_build.add_argument(
        '--variant',
        choices=['default', 'cublas', 'handmade', 'both'],
        default='default',
        help='Native library output variant to build',
    )

    sub.add_parser('prepare-data', help='Download and extract CIFAR-10 Python batches')
    sub.add_parser('info', help='Show important project paths and summary')
    sub.add_parser('doctor', help='Run first-run diagnostics for paths, data, and native CUDA')
    sub.add_parser('healthcheck', help='Validate framework wiring and native artifacts')
    sub.add_parser('list-flex-components', help='List configurable built-in components')
    sub.add_parser('list-dual-components', help='List dual-backend components and cuda_legacy subset')
    sub.add_parser('config-template', help='Print the PyTorch-flex config template')
    sub.add_parser('dual-config-template', help='Print the dual-backend unified config template')

    p_flex = sub.add_parser('train-flex', help='Train a configurable PyTorch model from YAML')
    p_flex.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    p_flex.add_argument('overrides', nargs='*', help='Overrides like train.epochs=2 optimizer.lr=0.001')

    p_dual = sub.add_parser('train-dual', help='Train with one shared model config and switch backend via engine.backend')
    p_dual.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_dual.add_argument('overrides', nargs='*', help='Overrides like engine.backend=cuda_legacy train.epochs=2')

    p_train_cuda = sub.add_parser('train-cuda', help='Alias for train-dual with engine.backend=cuda_legacy')
    p_train_cuda.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_train_cuda.add_argument('overrides', nargs='*')

    p_train_torch = sub.add_parser('train-torch', help='Alias for train-dual with engine.backend=torch')
    p_train_torch.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_train_torch.add_argument('overrides', nargs='*')

    p_compare = sub.add_parser('compare', help='Run backend comparison with one shared config')
    p_compare.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_compare.add_argument('--backends', nargs='+', default=['torch', 'cuda_legacy'], choices=['torch', 'cuda_legacy'])
    p_compare.add_argument('overrides', nargs='*')

    p_validate = sub.add_parser('validate-dual-config', help='Validate whether a config can run on cuda_legacy')
    p_validate.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate.add_argument('overrides', nargs='*')

    p_map = sub.add_parser('show-cuda-mapping', help='Show how a unified config maps onto the handcrafted CUDA backend')
    p_map.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_map.add_argument('overrides', nargs='*')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'build':
        build_native(
            use_cublas=not args.no_cublas,
            generator=args.generator,
            legacy_make=args.legacy_make,
            variant=args.variant,
        )
        if args.check:
            check_native(args.variant)
        return 0

    if args.command == 'prepare-data':
        from minicnn.data.prepare_cifar10 import main as prepare_data_main
        prepare_data_main()
        return 0

    if args.command == 'info':
        from minicnn.config import settings
        from minicnn.core.cuda_backend import resolve_library_path
        print(f'PROJECT_ROOT={PROJECT_ROOT}')
        print(f'CPP_ROOT={CPP_ROOT}')
        print(f'DATA_ROOT={DATA_ROOT}')
        print(f'CUDA_LIBRARY={resolve_library_path()}')
        print(f'Native library present={check_native()}')
        print('Resolved legacy settings:')
        print(json.dumps(settings.summarize(), indent=2))
        print(f'Health={healthcheck()}')
        print('Flexible registries:')
        print(json.dumps(describe_registries(), indent=2))
        print('Dual-backend handcrafted CUDA supported subset:')
        print(json.dumps(CUDA_LEGACY_SUPPORTED, indent=2))
        return 0

    if args.command == 'healthcheck':
        print(healthcheck())
        return 0

    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=2, default=str))
        return 0

    if args.command == 'list-flex-components':
        print(json.dumps(describe_registries(), indent=2))
        return 0

    if args.command == 'list-dual-components':
        print(json.dumps({'registries': describe_registries(), 'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED}, indent=2))
        return 0

    if args.command == 'config-template':
        print(dump_template())
        return 0

    if args.command == 'dual-config-template':
        print(dump_unified_template())
        return 0

    if args.command == 'train-flex':
        cfg = _load_flex_config_or_exit(args.config, args.overrides)
        run_dir = train_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'validate-dual-config':
        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        errors = validate_cuda_legacy_compatibility(cfg)
        if errors:
            print(json.dumps({'ok': False, 'errors': errors}, indent=2))
            return 2
        print(json.dumps({'ok': True, 'backend': 'cuda_legacy'}, indent=2))
        return 0

    if args.command == 'show-cuda-mapping':
        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        print(json.dumps(summarize_legacy_mapping(cfg), indent=2))
        return 0

    if args.command == 'train-dual':
        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command in {'train-cuda', 'train-torch'}:
        backend = 'cuda_legacy' if args.command == 'train-cuda' else 'torch'
        cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *args.overrides])
        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'compare':
        rows = []
        for backend in args.backends:
            t0 = time.perf_counter()
            cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *args.overrides])
            run_dir = train_unified_from_config(cfg)
            elapsed = time.perf_counter() - t0
            summary_path = run_dir / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
            rows.append({
                'backend': backend,
                'run_dir': str(run_dir),
                'best_model_path': summary.get('best_model_path'),
                'test_acc': summary.get('test_acc'),
                'elapsed_s': round(elapsed, 3),
            })
        print(json.dumps({'runs': rows}, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
