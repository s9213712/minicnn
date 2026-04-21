"""Tests for cuda_native debug / observability helpers."""
from __future__ import annotations

import numpy as np
import pytest

from minicnn.cuda_native.graph import build_graph
from minicnn.cuda_native.planner import make_naive_plan
from minicnn.cuda_native.debug import (
    dump_graph, dump_plan, inspect,
    TracingForwardExecutor, ExecutionTrace,
)


def _small_graph():
    layers = [
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 4},
        {'type': 'ReLU'},
    ]
    return build_graph(layers, input_shape=(2, 1, 8, 8))


def _small_params(graph):
    params = {}
    for node in graph.nodes:
        if node.op_type == 'Linear':
            in_f = node.input_specs[0].shape[1]
            out_f = node.output_specs[0].shape[1]
            params[f'_w_{node.name}'] = np.zeros((out_f, in_f), dtype=np.float32)
            params[f'_b_{node.name}'] = np.zeros(out_f, dtype=np.float32)
    return params


def _smoke_graph():
    return build_graph(
        [
            {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 3},
        ],
        input_shape=(1, 1, 5, 5),
    )


def test_dump_graph_runs_without_error():
    graph = _smoke_graph()
    out = dump_graph(graph)
    assert isinstance(out, str)
    assert len(out) > 0


def test_dump_plan_runs_without_error():
    graph = _smoke_graph()
    plan = make_naive_plan(graph)
    out = dump_plan(plan)
    assert isinstance(out, str)
    assert len(out) > 0


class TestDumpGraph:
    def test_returns_string(self):
        g = _small_graph()
        out = dump_graph(g)
        assert isinstance(out, str)

    def test_contains_node_names(self):
        g = _small_graph()
        out = dump_graph(g)
        for node in g.nodes:
            assert node.name in out

    def test_contains_op_types(self):
        g = _small_graph()
        out = dump_graph(g)
        for node in g.nodes:
            assert node.op_type in out

    def test_contains_input_shape(self):
        g = _small_graph()
        out = dump_graph(g)
        assert 'NativeGraph' in out
        assert 'input=' in out

    def test_shows_all_three_nodes(self):
        g = _small_graph()
        out = dump_graph(g)
        assert out.count('[') >= 3

    def test_empty_graph_does_not_crash(self):
        from minicnn.cuda_native.graph import NativeGraph
        g = NativeGraph()
        out = dump_graph(g)
        assert 'NativeGraph' in out


class TestDumpPlan:
    def test_returns_string(self):
        g = _small_graph()
        plan = make_naive_plan(g)
        out = dump_plan(plan)
        assert isinstance(out, str)

    def test_contains_steps(self):
        g = _small_graph()
        plan = make_naive_plan(g)
        out = dump_plan(plan)
        assert 'step' in out

    def test_contains_buffer_section(self):
        g = _small_graph()
        plan = make_naive_plan(g)
        out = dump_plan(plan)
        assert 'Buffers' in out

    def test_contains_total_kb(self):
        g = _small_graph()
        plan = make_naive_plan(g)
        out = dump_plan(plan)
        assert 'KB' in out

    def test_step_count_matches_nodes(self):
        g = _small_graph()
        plan = make_naive_plan(g)
        out = dump_plan(plan)
        assert out.count('step') >= len(g.nodes)


class TestInspect:
    def test_graph_only(self):
        g = _small_graph()
        out = inspect(g)
        assert 'NativeGraph' in out

    def test_graph_and_plan(self):
        g = _small_graph()
        plan = make_naive_plan(g)
        out = inspect(g, plan)
        assert 'NativeGraph' in out
        assert 'ExecutionPlan' in out


class TestTracingForwardExecutor:
    def test_returns_ctx_and_trace(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        ctx, trace = executor.run(g, {'input': x}, params)
        assert isinstance(trace, ExecutionTrace)
        assert isinstance(ctx, dict)

    def test_trace_has_correct_step_count(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        assert len(trace.node_traces) == len(g.nodes)

    def test_trace_node_names_match(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        for i, node in enumerate(g.nodes):
            assert trace.node_traces[i].node_name == node.name

    def test_trace_elapsed_ms_positive(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        for t in trace.node_traces:
            assert t.elapsed_ms >= 0.0

    def test_trace_output_shapes_populated(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        for t in trace.node_traces:
            assert len(t.output_shapes) > 0

    def test_trace_total_ms_is_sum(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        assert abs(trace.total_ms() - sum(t.elapsed_ms for t in trace.node_traces)) < 1e-9

    def test_trace_dump_is_string(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        assert isinstance(trace.dump(), str)

    def test_trace_summary_has_steps(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        _, trace = executor.run(g, {'input': x}, params)
        summary = trace.summary()
        assert 'steps' in summary
        assert len(summary['steps']) == len(g.nodes)

    def test_ctx_contains_output(self):
        g = _small_graph()
        params = _small_params(g)
        executor = TracingForwardExecutor()
        x = np.zeros((2, 1, 8, 8), dtype=np.float32)
        ctx, _ = executor.run(g, {'input': x}, params)
        assert g.output_spec.name in ctx
