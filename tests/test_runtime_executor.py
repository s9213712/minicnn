from __future__ import annotations

import numpy as np
import pytest

from minicnn.runtime.executor import GraphExecutor
from minicnn.runtime.graph import Graph, Node


class _IdentityModule:
    def __init__(self, **_attrs):
        pass

    def __call__(self, x):
        return x


def test_graph_executor_rejects_unsupported_multi_input_node(monkeypatch):
    graph = Graph()
    graph.add_node(Node(name='left', op='Identity', inputs=['input']))
    graph.add_node(Node(name='right', op='Identity', inputs=['input']))
    graph.add_node(Node(name='merge', op='Linear', inputs=['left', 'right'], attrs={'out_features': 2}))

    monkeypatch.setattr(
        'minicnn.runtime.executor.get_model_component',
        lambda _op: _IdentityModule,
    )

    executor = GraphExecutor(graph)

    with pytest.raises(NotImplementedError, match='Add/Concat'):
        executor.run([[1.0, 2.0, 3.0]])


def test_graph_executor_supports_add_multi_input_node():
    graph = Graph()
    graph.add_node(Node(name='left', op='ReLU', inputs=['input']))
    graph.add_node(Node(name='right', op='ReLU', inputs=['input']))
    graph.add_node(Node(name='sum', op='Add', inputs=['left', 'right']))

    executor = GraphExecutor(graph)
    values = executor.run([[1.0, 2.0, 3.0]])

    np.testing.assert_allclose(values['sum'].data, np.array([[2.0, 4.0, 6.0]], dtype=np.float32))


def test_graph_executor_supports_concat_multi_input_node():
    graph = Graph()
    graph.add_node(Node(name='left', op='ReLU', inputs=['input']))
    graph.add_node(Node(name='right', op='ReLU', inputs=['input']))
    graph.add_node(Node(name='cat', op='Concat', inputs=['left', 'right'], attrs={'axis': 1}))

    executor = GraphExecutor(graph)
    values = executor.run([[1.0, 2.0, 3.0]])

    np.testing.assert_allclose(
        values['cat'].data,
        np.array([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]], dtype=np.float32),
    )

