from __future__ import annotations

import math

import numpy as np

import minicnn.training.cuda_batch as cuda_batch


def test_global_clip_scale_disabled_or_under_limit():
    assert cuda_batch.global_clip_scale(total_grad_sq=100.0, max_norm=0.0) == 1.0
    assert cuda_batch.global_clip_scale(total_grad_sq=4.0, max_norm=2.0) == 1.0
    assert cuda_batch.global_clip_scale(total_grad_sq=1.0, max_norm=-1.0) == 1.0


def test_global_clip_scale_scales_down_by_norm():
    scale = cuda_batch.global_clip_scale(total_grad_sq=25.0, max_norm=2.0)
    assert math.isclose(scale, 0.4, rel_tol=1e-9)


def test_scaled_normalizer_applies_clip_scale_through_existing_update_api():
    assert cuda_batch.scaled_normalizer(8.0, 1.0) == 8.0
    assert cuda_batch.scaled_normalizer(8.0, 0.25) == 32.0


def test_cuda_grad_clip_plan_reuses_shared_normalizer_logic():
    plan = cuda_batch.CudaGradClipPlan(global_scale=0.25, conv_normalizers=(8.0, 2.0))

    assert plan.fc_normalizer() == 4.0
    assert plan.conv_normalizer(0) == 32.0
    assert plan.conv_normalizer(1) == 8.0


def test_make_grad_clip_plan_uses_conv_normalizers_and_global_scale(monkeypatch):
    expected_arch = object()
    expected_workspace = object()
    calls: list[tuple[object, object, float, list[float]]] = []

    monkeypatch.setattr(cuda_batch, 'conv_grad_normalizers', lambda arch: [4.0, 8.0] if arch is expected_arch else [])

    def fake_global_scale(workspace, arch, max_norm, normalizers):
        calls.append((workspace, arch, max_norm, list(normalizers)))
        return 0.5

    monkeypatch.setattr(cuda_batch, 'cuda_global_grad_scale', fake_global_scale)

    plan = cuda_batch.make_grad_clip_plan(expected_workspace, expected_arch, max_norm=3.0)

    assert plan == cuda_batch.CudaGradClipPlan(global_scale=0.5, conv_normalizers=(4.0, 8.0))
    assert calls == [(expected_workspace, expected_arch, 3.0, [4.0, 8.0])]


def test_device_grad_sq_uses_normalized_gradient(monkeypatch):
    monkeypatch.setattr(cuda_batch, 'g2h', lambda _ptr, _size: np.array([2.0, 4.0], dtype=np.float32))
    monkeypatch.setattr(cuda_batch, 'is_lib_loaded', lambda: True)

    assert cuda_batch.device_grad_sq(object(), 2, normalizer=2.0) == 5.0
