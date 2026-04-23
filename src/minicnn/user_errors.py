from __future__ import annotations


def format_user_error(
    problem: str,
    *,
    cause: str | None = None,
    fix: str | None = None,
    example: str | None = None,
) -> str:
    lines = [f'[ERROR] {problem}']
    if cause:
        lines.append(f'-> Cause: {cause}')
    if fix:
        lines.append(f'-> Fix: {fix}')
    if example:
        lines.append('-> Example:')
        for line in str(example).splitlines():
            lines.append(f'   {line}')
    return '\n'.join(lines)


def format_dataset_split_error(
    *,
    dataset_name: str,
    train_pool_size: int,
    num_samples: int,
    val_samples: int,
    example_num_samples: int,
    example_val_samples: int,
) -> str:
    return format_user_error(
        'Dataset split invalid',
        cause=(
            'num_samples + val_samples exceeds available training samples '
            f'for {dataset_name}: {num_samples} + {val_samples} > {train_pool_size}'
        ),
        fix=(
            'Reduce num_samples or val_samples so the train/validation split fits '
            'inside the training pool.'
        ),
        example=(
            f'num_samples={example_num_samples}\n'
            f'val_samples={example_val_samples}'
        ),
    )
