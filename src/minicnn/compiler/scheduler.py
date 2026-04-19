from __future__ import annotations


def schedule(graph):
    return {'nodes': graph.topological_order(), 'buffer_lifetimes': {}}
