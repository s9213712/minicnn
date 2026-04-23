from __future__ import annotations

from minicnn._cli_config import (
    _resolve_cli_config_path,
)
from minicnn._cli_parser import build_cli_parser
from minicnn._cli_readonly import (
    handle_config_template,
    handle_doctor,
    handle_evaluate_checkpoint,
    handle_dual_config_template,
    handle_export_torch_checkpoint,
    handle_healthcheck,
    handle_info,
    handle_inspect_checkpoint,
    handle_list_dual_components,
    handle_list_flex_components,
    handle_show_cuda_mapping,
    handle_smoke,
    handle_validate_dual_config,
)
from minicnn._cli_introspection import (
    handle_compile,
    handle_show_graph,
    handle_show_model,
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
from minicnn._cli_validation import (
    handle_cuda_native_capabilities,
    handle_validate_config,
    handle_validate_cuda_native_config,
)
from minicnn.core.build import build_native, check_native


def build_parser():
    return build_cli_parser()


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

    if args.command == 'evaluate-checkpoint':
        return handle_evaluate_checkpoint(args)

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
