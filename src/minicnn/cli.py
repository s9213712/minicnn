from __future__ import annotations

import argparse

from minicnn._cli_config import (
    _resolve_cli_config_path,
)
from minicnn._cli_output import (
    _add_format_arg,
)
from minicnn._cli_readonly import (
    handle_compile,
    handle_config_template,
    handle_cuda_native_capabilities,
    handle_doctor,
    handle_dual_config_template,
    handle_export_torch_checkpoint,
    handle_healthcheck,
    handle_info,
    handle_inspect_checkpoint,
    handle_list_dual_components,
    handle_list_flex_components,
    handle_show_cuda_mapping,
    handle_show_graph,
    handle_show_model,
    handle_smoke,
    handle_validate_config,
    handle_validate_cuda_native_config,
    handle_validate_dual_config,
)
from minicnn._cli_training import (
    benchmark_fields as _benchmark_fields,
    common_train_overrides as _common_train_overrides,
    compare_backends_and_overrides as _compare_backends_and_overrides,
    handle_compare,
    handle_train_autograd,
    handle_train_dual,
    handle_train_dual_alias,
    handle_train_flex,
    handle_train_native,
)
from minicnn.core.build import build_native, check_native


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--lr-conv1', type=float)
    parser.add_argument('--lr-conv', type=float)
    parser.add_argument('--lr-fc', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--dataset-seed', type=int)
    parser.add_argument('--init-seed', type=int)
    parser.add_argument('--train-seed', type=int)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--eval-max-batches', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--grad-debug', action='store_true')
    parser.add_argument('--grad-debug-batches', type=int)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='minicnn', description='MiniCNN dual-backend CLI (pure handcrafted CUDA + PyTorch)')
    sub = parser.add_subparsers(dest='command', required=True)

    p_build = sub.add_parser('build', help='Build CUDA shared library')
    p_build.add_argument('--no-cublas', action='store_true', help='Disable cuBLAS and use handwritten fallback')
    p_build.add_argument('--check', action='store_true', help='Run symbol/export checks after build')
    p_build.add_argument('--generator', choices=['make', 'ninja'], default='make')
    p_build.add_argument('--legacy-make', action='store_true', help='Use original cpp/Makefile instead of CMake')
    p_build.add_argument('--cuda-arch', default='86', help='CUDA architecture, for example 86 or sm_86')
    p_build.add_argument(
        '--variant',
        choices=['default', 'cublas', 'handmade', 'both'],
        default='default',
        help='Native library output variant to build',
    )

    sub.add_parser('prepare-data', help='Download and extract CIFAR-10 Python batches')
    p_info = sub.add_parser('info', help='Show important project paths and summary')
    _add_format_arg(p_info)
    p_doctor = sub.add_parser('doctor', help='Run first-run diagnostics for paths, data, and native CUDA')
    _add_format_arg(p_doctor)
    p_health = sub.add_parser('healthcheck', help='Validate framework wiring and native artifacts')
    _add_format_arg(p_health)
    p_smoke = sub.add_parser('smoke', help='Run a compact first-run self-check across configs, compiler, and backend validation')
    _add_format_arg(p_smoke)
    sub.add_parser('list-flex-components', help='List configurable built-in components')
    sub.add_parser('list-dual-components', help='List dual-backend components and cuda_legacy subset')
    sub.add_parser('config-template', help='Print the PyTorch-flex config template')
    sub.add_parser('dual-config-template', help='Print the dual-backend unified config template')
    p_inspect_ckpt = sub.add_parser('inspect-checkpoint', help='Inspect a saved model/checkpoint artifact (.pt/.pth/.npz)')
    p_inspect_ckpt.add_argument('--path', required=True, type=str)
    _add_format_arg(p_inspect_ckpt)
    p_export_ckpt = sub.add_parser('export-torch-checkpoint', help='Export a supported MiniCNN checkpoint to a generic torch checkpoint (.pt)')
    p_export_ckpt.add_argument('--path', required=True, type=str)
    p_export_ckpt.add_argument('--config', required=True, type=str)
    p_export_ckpt.add_argument('--output', required=True, type=str)

    p_flex = sub.add_parser('train-flex', help='Train a configurable PyTorch model from YAML')
    p_flex.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    _add_common_train_args(p_flex)
    p_flex.add_argument('overrides', nargs='*', help='Overrides like train.epochs=2 optimizer.lr=0.001')

    p_dual = sub.add_parser('train-dual', help='Train with one shared model config and switch backend via engine.backend')
    p_dual.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_dual)
    p_dual.add_argument('overrides', nargs='*', help='Overrides like engine.backend=cuda_legacy train.epochs=2')

    p_train = sub.add_parser('train', help='Alias for train-dual')
    p_train.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_train)
    p_train.add_argument('overrides', nargs='*')

    p_train_cuda = sub.add_parser('train-cuda', help='Alias for train-dual with engine.backend=cuda_legacy')
    p_train_cuda.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_train_cuda)
    p_train_cuda.add_argument('overrides', nargs='*')

    p_train_torch = sub.add_parser('train-torch', help='Alias for train-dual with engine.backend=torch')
    p_train_torch.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_train_torch)
    p_train_torch.add_argument('overrides', nargs='*')

    p_train_autograd = sub.add_parser('train-autograd', help='Train the CPU/NumPy MiniCNN autograd path')
    p_train_autograd.add_argument('--config', type=str, default='configs/autograd_tiny.yaml')
    _add_common_train_args(p_train_autograd)
    p_train_autograd.add_argument('overrides', nargs='*')

    p_compare = sub.add_parser('compare', help='Run backend comparison with one shared config')
    p_compare.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_compare.add_argument('--backends', nargs='+', default=None, metavar='BACKEND')
    p_compare.add_argument('--backend-a', choices=['torch', 'cuda_legacy', 'autograd'])
    p_compare.add_argument('--backend-b', choices=['torch', 'cuda_legacy', 'autograd'])
    _add_common_train_args(p_compare)
    p_compare.add_argument('overrides', nargs='*')

    p_validate = sub.add_parser('validate-dual-config', help='Validate whether a config can run on cuda_legacy')
    p_validate.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate.add_argument('overrides', nargs='*')
    _add_format_arg(p_validate)

    p_map = sub.add_parser('show-cuda-mapping', help='Show how a unified config maps onto the handcrafted CUDA backend')
    p_map.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_map.add_argument('overrides', nargs='*')
    _add_format_arg(p_map)

    p_validate_any = sub.add_parser('validate-config', help='Validate shared config shape and backend compatibility')
    p_validate_any.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate_any.add_argument('overrides', nargs='*')
    _add_format_arg(p_validate_any)

    p_compile = sub.add_parser('compile', help='Trace and optimize a model config into MiniCNN IR')
    p_compile.add_argument('--config', type=str, default='configs/autograd_tiny.yaml')
    p_compile.add_argument('overrides', nargs='*')
    p_show_model = sub.add_parser('show-model', help='Render a human-oriented model summary from config/frontend structure')
    p_show_model.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    p_show_model.add_argument('overrides', nargs='*')
    _add_format_arg(p_show_model)
    p_show_graph = sub.add_parser('show-graph', help='Render the canonical primitive graph traced from config')
    p_show_graph.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    p_show_graph.add_argument('overrides', nargs='*')
    _add_format_arg(p_show_graph)

    p_train_native = sub.add_parser('train-native', help='[EXPERIMENTAL] Train with cuda_native backend (research prototype)')
    p_train_native.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_train_native.add_argument('overrides', nargs='*')

    p_validate_native = sub.add_parser('validate-cuda-native-config', help='[EXPERIMENTAL] Validate config against cuda_native constraints')
    p_validate_native.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate_native.add_argument('overrides', nargs='*')
    _add_format_arg(p_validate_native)

    sub.add_parser('cuda-native-capabilities', help='[EXPERIMENTAL] Print cuda_native capability descriptor')

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
            cuda_arch=args.cuda_arch,
        )
        if args.check:
            check_native(args.variant)
        return 0

    if args.command == 'prepare-data':
        from minicnn.data.prepare_cifar10 import main as prepare_data_main
        prepare_data_main()
        return 0

    if args.command == 'info':
        return handle_info(args)

    if args.command == 'healthcheck':
        return handle_healthcheck(args)

    if args.command == 'smoke':
        return handle_smoke(args)

    if args.command == 'doctor':
        return handle_doctor(args)

    if args.command == 'list-flex-components':
        return handle_list_flex_components()

    if args.command == 'list-dual-components':
        return handle_list_dual_components()

    if args.command == 'config-template':
        return handle_config_template()

    if args.command == 'dual-config-template':
        return handle_dual_config_template()

    if args.command == 'inspect-checkpoint':
        return handle_inspect_checkpoint(args)

    if args.command == 'export-torch-checkpoint':
        return handle_export_torch_checkpoint(args)

    if args.command == 'train-flex':
        return handle_train_flex(args)

    if args.command == 'validate-dual-config':
        return handle_validate_dual_config(args)

    if args.command == 'show-cuda-mapping':
        return handle_show_cuda_mapping(args)

    if args.command in {'train', 'train-dual'}:
        return handle_train_dual(args)

    if args.command in {'train-cuda', 'train-torch'}:
        return handle_train_dual_alias(args)

    if args.command == 'train-autograd':
        return handle_train_autograd(args)

    if args.command == 'compare':
        return handle_compare(args, parser)

    if args.command == 'validate-config':
        return handle_validate_config(args)

    if args.command == 'validate-cuda-native-config':
        return handle_validate_cuda_native_config(args)

    if args.command == 'cuda-native-capabilities':
        return handle_cuda_native_capabilities()

    if args.command == 'train-native':
        return handle_train_native(args)

    if args.command == 'compile':
        return handle_compile(args)

    if args.command == 'show-model':
        return handle_show_model(args)

    if args.command == 'show-graph':
        return handle_show_graph(args)

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
