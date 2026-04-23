from __future__ import annotations

from minicnn._cli_config import _load_flex_config_or_exit
from minicnn._cli_output import _print_graph_view, _print_json, _print_model_view


def handle_compile(args) -> int:
    from minicnn.compiler import optimize, trace_model_config

    cfg = _load_flex_config_or_exit(args.config, args.overrides)
    graph = optimize(trace_model_config(cfg.get('model', {})))
    _print_json({
        'command': 'compile',
        'schema_version': 1,
        'kind': 'compiled_graph_summary',
        'status': 'ok',
        **graph.summary(),
    })
    return 0


def handle_show_model(args) -> int:
    from minicnn.introspection.model_view import build_model_view_from_config, render_model_view_text

    cfg = _load_flex_config_or_exit(args.config, args.overrides)
    view = build_model_view_from_config(cfg)
    payload = {
        'status': 'ok',
        'schema_version': 1,
        'kind': 'model_view',
        'model_type': view.model_type,
        'input_shape': view.input_shape,
        'backend_intent': view.backend_intent,
        'summary': view.summary,
        'layers': [layer.to_dict() for layer in view.layers],
        'text': render_model_view_text(view),
    }
    _print_model_view(payload, command='show-model', output_format=args.format)
    return 0


def handle_show_graph(args) -> int:
    from minicnn.introspection.graph_view import build_graph_view_from_config, render_graph_view_text

    cfg = _load_flex_config_or_exit(args.config, args.overrides)
    payload = {
        'status': 'ok',
        'schema_version': 1,
        'kind': 'graph_view',
        **build_graph_view_from_config(cfg),
    }
    payload['text'] = render_graph_view_text(payload)
    _print_graph_view(payload, command='show-graph', output_format=args.format)
    return 0
