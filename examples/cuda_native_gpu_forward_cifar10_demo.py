"""Run a partial native-forward cuda_native GPU smoke on a CIFAR-10 batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minicnn.cuda_native import ForwardExecutor, build_cuda_native_graph, make_native_gpu_forward_executor
from minicnn.data.cifar10 import cifar10_ready, load_cifar10_test, normalize_cifar


def _build_graph(batch_size: int):
    return build_cuda_native_graph(
        {
            "layers": [
                {"type": "Conv2d", "out_channels": 4, "kernel_size": 3, "padding": 0, "bias": False},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "kernel_size": 2, "stride": 2},
                {"type": "Flatten"},
                {"type": "Linear", "out_features": 10},
            ],
        },
        (batch_size, 3, 32, 32),
    )


def _init_demo_params(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "_w_conv2d_0": rng.normal(0.0, 0.03, size=(4, 3, 3, 3)).astype(np.float32),
        "_w_linear_4": rng.normal(0.0, 0.02, size=(10, 4 * 15 * 15)).astype(np.float32),
        "_b_linear_4": np.zeros((10,), dtype=np.float32),
    }


def _load_batch(data_root: str, batch_size: int, *, allow_random_fallback: bool) -> tuple[np.ndarray, str]:
    root = Path(data_root)
    if cifar10_ready(root):
        x_test, _y_test = load_cifar10_test(root, download=False)
        return normalize_cifar(x_test[:batch_size]).astype(np.float32), "official:cifar10:test_batch"
    if not allow_random_fallback:
        raise FileNotFoundError(
            f"CIFAR-10 data is missing under {root}. Run `minicnn prepare-data` or pass "
            "`--allow-random-fallback` for a deterministic non-dataset smoke."
        )
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(batch_size, 3, 32, 32)).astype(np.float32), "random:fallback"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cuda_native partial native-forward GPU execution on a CIFAR-10 batch."
    )
    parser.add_argument("--data-root", default="data/cifar-10-batches-py")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-random-fallback", action="store_true")
    args = parser.parse_args()

    batch_size = max(1, int(args.batch_size))
    graph = _build_graph(batch_size)
    params = _init_demo_params(args.seed)
    x, dataset_source = _load_batch(
        args.data_root,
        batch_size,
        allow_random_fallback=bool(args.allow_random_fallback),
    )

    native_executor = make_native_gpu_forward_executor(reserve_bytes=64 * 1024 * 1024, reserve_buffers=8)
    native_result = native_executor.run(graph, x, params=params)
    reference_output = ForwardExecutor().run_inference(graph, x, params=params)
    max_abs_diff = float(np.max(np.abs(native_result.output - reference_output)))

    runtime_summary = native_executor.device_runtime.summary()
    payload = {
        "demo": "cuda_native_gpu_forward_cifar10",
        "dataset_source": dataset_source,
        "batch_shape": list(x.shape),
        "output_shape": list(native_result.output.shape),
        "max_abs_diff_vs_reference_numpy": max_abs_diff,
        "native_execution_kinds": {
            key: value
            for key, value in runtime_summary["execution_kinds"].items()
            if key.startswith("gpu_native_")
        },
        "device_pointer_allocation_events": runtime_summary["device_pointer_allocation_events"],
        "device_sync_to_device_events": runtime_summary["device_sync_to_device_events"],
        "device_sync_to_host_events": runtime_summary["device_sync_to_host_events"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
