from __future__ import annotations


def lower(graph, backend: str = 'numpy') -> dict[str, object]:
    if backend not in {'numpy', 'torch', 'cuda'}:
        raise ValueError(f'Unknown backend for lowering: {backend}')
    if backend == 'cuda':
        return {'backend': backend, 'supported': False, 'reason': 'direct CUDA graph lowering is not implemented; use cuda_legacy trainer'}
    return {'backend': backend, 'supported': True, 'nodes': graph.topological_order()}
