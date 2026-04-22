from __future__ import annotations

import argparse
import json
from typing import Any


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _add_format_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--format', choices=['json', 'text'], default='json')


def _text_block(title: str, payload: Any) -> str:
    rendered = json.dumps(payload, indent=2, default=str)
    return f'{title}:\n{rendered}'


def _print_diagnostic(payload: dict[str, Any], *, command: str, output_format: str) -> None:
    if output_format == 'json':
        enriched = {'command': command, **payload}
        _print_json(enriched)
        return

    status = payload.get('status', 'ok')
    checks = payload.get('checks', [])
    warnings = payload.get('warnings', [])
    errors = payload.get('errors', [])
    print(f'{command}: {status}')
    if checks:
        for check in checks:
            marker = 'ok' if check.get('ok') else ('warn' if not check.get('required') else 'error')
            print(f"- [{marker}] {check.get('name')}")
            suggested_fix = check.get('suggested_fix')
            if suggested_fix and not check.get('ok'):
                print(f"  fix: {suggested_fix}")
    if warnings:
        print(_text_block('warnings', warnings))
    if errors:
        print(_text_block('errors', errors))


def _print_validation_result(payload: dict[str, Any], *, command: str, output_format: str) -> None:
    if output_format == 'json':
        _print_json({'command': command, **payload})
        return

    status = payload.get('status', 'ok' if payload.get('ok', False) else 'error')
    print(f'{command}: {status}')
    backend = payload.get('backend')
    if backend:
        print(f'backend: {backend}')
    note = payload.get('note')
    if note:
        print(f'note: {note}')
    errors = payload.get('errors', [])
    if errors:
        print(_text_block('errors', errors))


def _print_generic_payload(payload: dict[str, Any], *, command: str, output_format: str) -> None:
    if output_format == 'json':
        _print_json({'command': command, **payload})
        return

    headline = payload.get('status') or payload.get('kind') or payload.get('format') or 'ok'
    print(f'{command}: {headline}')
    for key in ('path', 'format', 'kind', 'backend', 'note'):
        value = payload.get(key)
        if value not in (None, '', [], {}):
            print(f'{key}: {value}')
    for key in (
        'compatible_backends',
        'metadata',
        'top_level_keys',
        'state_keys',
        'keys',
        'preview',
        'project',
        'model',
        'optim',
        'train',
        'runtime',
        'warnings',
        'errors',
    ):
        value = payload.get(key)
        if value not in (None, '', [], {}):
            print(_text_block(key, value))
