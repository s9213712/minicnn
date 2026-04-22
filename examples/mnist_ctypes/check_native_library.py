#!/usr/bin/env python3
"""Minimal native-library smoke test for MiniCNN's handcrafted CUDA backend.

This script is intentionally small and cross-platform:
- Linux: loads `.so`
- Windows: loads `.dll`

It validates that the resolved native library is present, loadable through the
repo's ctypes bindings, exports the required symbols, and can complete a simple
GPU upload/download round-trip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from minicnn.core.cuda_backend import check_cuda_ready, get_lib, reset_library_cache
from minicnn.core._cuda_library import resolve_library_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test MiniCNN native CUDA library loading.")
    parser.add_argument(
        "--path",
        help="Explicit native library path, or a path under cpp/. Example: cpp/minimal_cuda_cnn_cublas.dll",
    )
    parser.add_argument(
        "--variant",
        choices=["default", "cublas", "handmade", "nocublas"],
        help="Resolve a built-in native variant name through the repo loader.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of float32 values to round-trip through GPU memory.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def _resolve_target(args: argparse.Namespace) -> str:
    target = args.path or args.variant
    return resolve_library_path(target)


def _run_roundtrip(count: int) -> dict[str, object]:
    lib = get_lib()
    host = np.linspace(0.0, 1.0, count, dtype=np.float32)
    ptr = lib.gpu_malloc(host.nbytes)
    if not ptr:
        raise RuntimeError(f"gpu_malloc failed for {host.nbytes} bytes")

    try:
        lib.gpu_memcpy_h2d(ptr, host.ctypes.data, host.nbytes)
        out = np.empty_like(host)
        lib.gpu_memcpy_d2h(out.ctypes.data, ptr, out.nbytes)
        if hasattr(lib, "gpu_synchronize"):
            lib.gpu_synchronize()
    finally:
        lib.gpu_free(ptr)

    return {
        "count": count,
        "bytes": int(host.nbytes),
        "roundtrip_ok": bool(np.array_equal(host, out)),
        "preview": out[: min(8, count)].tolist(),
    }


def main() -> int:
    args = _parse_args()
    resolved = _resolve_target(args)

    os.environ["MINICNN_CUDA_SO"] = resolved
    os.environ.pop("MINICNN_CUDA_VARIANT", None)
    reset_library_cache()

    ready = check_cuda_ready(resolved)
    payload: dict[str, object] = {
        "status": "ok",
        "resolved_path": resolved,
        "ready": ready,
    }

    if not ready["loadable"]:
        payload["status"] = "error"
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(f"[ERROR] failed to load native library: {resolved}")
            print(json.dumps(ready, indent=2))
        return 1

    roundtrip = _run_roundtrip(args.count)
    payload["roundtrip"] = roundtrip
    if not roundtrip["roundtrip_ok"]:
        payload["status"] = "error"

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"Resolved native library: {resolved}")
        print("Required symbols: ok")
        print(f"GPU round-trip: {'ok' if roundtrip['roundtrip_ok'] else 'failed'}")
        print(f"Preview: {roundtrip['preview']}")

    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
