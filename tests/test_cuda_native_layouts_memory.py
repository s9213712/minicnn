"""Tests for cuda_native layouts.py and memory.py."""
from __future__ import annotations

import numpy as np
import pytest

from minicnn.cuda_native.graph import build_graph
from minicnn.cuda_native.planner import make_naive_plan
from minicnn.cuda_native.layouts import (
    NCHW, NC, C, NHWC, SCALAR,
    KNOWN_LAYOUTS, SUPPORTED_ACTIVATION_LAYOUTS,
    LayoutSpec, infer_layout,
    expected_input_layout, expected_output_layout,
    validate_op_layout, validate_graph_layouts,
    OP_LAYOUT_RULES,
)
from minicnn.cuda_native.memory import (
    BufferAllocator, BufferPool, memory_footprint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _graph():
    layers = [
        {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3},
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 2},
    ]
    return build_graph(layers, input_shape=(2, 1, 8, 8))


# ---------------------------------------------------------------------------
# layouts.py tests
# ---------------------------------------------------------------------------

class TestLayoutConstants:
    def test_known_layouts_contains_all_constants(self):
        for layout in (NCHW, NHWC, NC, C, SCALAR):
            assert layout in KNOWN_LAYOUTS

    def test_supported_activation_layouts(self):
        assert NCHW in SUPPORTED_ACTIVATION_LAYOUTS
        assert NC in SUPPORTED_ACTIVATION_LAYOUTS
        assert NHWC not in SUPPORTED_ACTIVATION_LAYOUTS

    def test_op_layout_rules_cover_supported_ops(self):
        for op in (
            'Add',
            'Conv2d',
            'DepthwiseConv2d',
            'PointwiseConv2d',
            'ResidualBlock',
            'ConvNeXtBlock',
            'BatchNorm2d',
            'LayerNorm2d',
            'ReLU',
            'LeakyReLU',
            'GELU',
            'Dropout',
            'MaxPool2d',
            'AvgPool2d',
            'AdaptiveAvgPool2d',
            'GlobalAvgPool2d',
            'Flatten',
            'Linear',
        ):
            assert op in OP_LAYOUT_RULES


class TestLayoutSpec:
    def test_rank_4d(self):
        spec = LayoutSpec(shape=(2, 4, 6, 6), layout=NCHW)
        assert spec.rank == 4

    def test_is_image_true(self):
        spec = LayoutSpec(shape=(2, 4, 6, 6), layout=NCHW)
        assert spec.is_image()

    def test_is_image_false_for_nc(self):
        spec = LayoutSpec(shape=(2, 10), layout=NC)
        assert not spec.is_image()

    def test_is_flat_true(self):
        spec = LayoutSpec(shape=(2, 64), layout=NC)
        assert spec.is_flat()

    def test_is_flat_false_for_nchw(self):
        spec = LayoutSpec(shape=(2, 4, 6, 6), layout=NCHW)
        assert not spec.is_flat()

    def test_validate_nchw_correct(self):
        spec = LayoutSpec(shape=(2, 4, 6, 6), layout=NCHW)
        assert spec.validate() == []

    def test_validate_nchw_wrong_rank(self):
        spec = LayoutSpec(shape=(2, 64), layout=NCHW)
        errors = spec.validate()
        assert len(errors) > 0
        assert 'rank' in errors[0]

    def test_validate_nc_correct(self):
        spec = LayoutSpec(shape=(2, 64), layout=NC)
        assert spec.validate() == []

    def test_validate_unknown_layout(self):
        spec = LayoutSpec(shape=(2, 4, 6, 6), layout='WEIRD')
        errors = spec.validate()
        assert len(errors) > 0


class TestInferLayout:
    def test_rank4_gives_nchw(self):
        assert infer_layout((2, 3, 32, 32)) == NCHW

    def test_rank2_gives_nc(self):
        assert infer_layout((2, 128)) == NC

    def test_rank1_gives_c(self):
        assert infer_layout((128,)) == C

    def test_rank0_gives_scalar(self):
        assert infer_layout(()) == SCALAR


class TestExpectedLayouts:
    def test_add_is_layout_passthrough_contract(self):
        assert expected_input_layout('Add') is None
        assert expected_output_layout('Add') is None

    def test_conv2d_expects_nchw(self):
        assert expected_input_layout('Conv2d') == NCHW
        assert expected_output_layout('Conv2d') == NCHW

    def test_flatten_expects_nchw_produces_nc(self):
        assert expected_input_layout('Flatten') == NCHW
        assert expected_output_layout('Flatten') == NC

    def test_linear_expects_nc(self):
        assert expected_input_layout('Linear') == NC

    def test_relu_passthrough(self):
        assert expected_input_layout('ReLU') is None
        assert expected_output_layout('ReLU') is None


class TestValidateOpLayout:
    def test_correct_layout_no_errors(self):
        errors = validate_op_layout('Conv2d', NCHW, 'conv_0')
        assert errors == []

    def test_wrong_layout_gives_error(self):
        errors = validate_op_layout('Conv2d', NC, 'conv_0')
        assert len(errors) == 1
        assert 'conv_0' in errors[0]

    def test_passthrough_op_accepts_any(self):
        assert validate_op_layout('ReLU', NCHW, 'relu_0') == []
        assert validate_op_layout('ReLU', NC, 'relu_0') == []


class TestValidateGraphLayouts:
    def test_valid_graph_no_errors(self):
        g = _graph()
        errors = validate_graph_layouts(g)
        assert errors == [], f'Unexpected errors: {errors}'

    def test_returns_list(self):
        g = _graph()
        assert isinstance(validate_graph_layouts(g), list)

    def test_add_graph_with_matching_layouts_is_valid(self):
        g = build_graph(
            [
                {'type': 'Identity', 'output': 'stem'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
                {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
            ],
            input_shape=(2, 3, 8, 8),
        )
        assert validate_graph_layouts(g) == []

    def test_add_graph_shape_mismatch_is_rejected_at_build_time(self):
        with pytest.raises(ValueError, match='Add expects all input shapes to match'):
            build_graph(
                [
                    {'type': 'Identity', 'output': 'image'},
                    {'type': 'Flatten', 'inputs': ['image'], 'output': 'flat'},
                    {'type': 'Add', 'inputs': ['image', 'flat'], 'output': 'sum'},
                ],
                input_shape=(2, 3, 8, 8),
            )


class TestPlannerWithAdd:
    def test_naive_plan_preserves_multi_input_step_buffers(self):
        g = build_graph(
            [
                {'type': 'Identity', 'output': 'stem'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
                {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
                {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
            ],
            input_shape=(2, 3),
        )
        plan = make_naive_plan(g)
        step = plan.steps[-1]
        assert step.op_type == 'Add'
        assert step.inputs == ['left', 'right']
        assert len(step.input_buffers) == 2
        assert step.output_buffers


# ---------------------------------------------------------------------------
# memory.py tests
# ---------------------------------------------------------------------------

class TestBufferAllocator:
    def test_from_plan_builds_allocator(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_plan(plan)
        assert allocator.num_buffers == plan.buffer_plan.num_buffers

    def test_from_tensor_specs_builds_allocator(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_tensor_specs(plan, g)
        assert allocator.num_buffers == plan.buffer_plan.num_buffers

    def test_allocate_returns_numpy_arrays(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_tensor_specs(plan, g)
        pool = allocator.allocate()
        assert all(isinstance(v, np.ndarray) for v in pool.values())

    def test_allocate_arrays_are_zeros(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_tensor_specs(plan, g)
        pool = allocator.allocate()
        for arr in pool.values():
            assert np.all(arr == 0)

    def test_allocate_float32_dtype(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_tensor_specs(plan, g)
        pool = allocator.allocate()
        for arr in pool.values():
            assert arr.dtype == np.float32

    def test_total_bytes_positive(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_tensor_specs(plan, g)
        assert allocator.total_bytes > 0

    def test_summary_keys(self):
        g = _graph()
        plan = make_naive_plan(g)
        allocator = BufferAllocator.from_tensor_specs(plan, g)
        s = allocator.summary()
        assert 'num_buffers' in s
        assert 'total_bytes' in s
        assert 'total_kb' in s
        assert 'dtype' in s


class TestBufferPool:
    def test_build_creates_pool(self):
        g = _graph()
        plan = make_naive_plan(g)
        pool = BufferPool.build(plan, g)
        assert pool.num_buffers > 0

    def test_total_bytes_positive(self):
        g = _graph()
        plan = make_naive_plan(g)
        pool = BufferPool.build(plan, g)
        assert pool.total_bytes > 0

    def test_reset_zeros_all_buffers(self):
        g = _graph()
        plan = make_naive_plan(g)
        pool = BufferPool.build(plan, g)
        pool._pool[list(pool._pool.keys())[0]][:] = 99.0
        pool.reset()
        for arr in pool._pool.values():
            assert np.all(arr == 0)

    def test_make_ctx_includes_feeds(self):
        g = _graph()
        plan = make_naive_plan(g)
        pool = BufferPool.build(plan, g)
        x = np.ones((2, 1, 8, 8), dtype=np.float32)
        ctx = pool.make_ctx(feeds={'input': x})
        assert 'input' in ctx
        assert np.all(ctx['input'] == 1.0)

    def test_make_ctx_includes_params(self):
        g = _graph()
        plan = make_naive_plan(g)
        pool = BufferPool.build(plan, g)
        params = {'_w_test': np.zeros((4, 1, 3, 3), dtype=np.float32)}
        ctx = pool.make_ctx(feeds={}, params=params)
        assert '_w_test' in ctx

    def test_summary_has_required_keys(self):
        g = _graph()
        plan = make_naive_plan(g)
        pool = BufferPool.build(plan, g)
        s = pool.summary()
        assert 'num_buffers' in s
        assert 'total_kb' in s


class TestMemoryFootprint:
    def test_returns_dict(self):
        g = _graph()
        fp = memory_footprint(g)
        assert isinstance(fp, dict)

    def test_has_total_bytes(self):
        g = _graph()
        fp = memory_footprint(g)
        assert fp['total_bytes'] > 0

    def test_num_buffers_matches_layers_plus_one(self):
        g = _graph()
        fp = memory_footprint(g)
        # naive plan: 1 input + 1 per node output
        assert fp['num_buffers'] == len(g.nodes) + 1
