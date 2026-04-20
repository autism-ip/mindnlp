"""
Phase 3: Single-operator performance comparison (Triton vs Native).

For each operator and each shape, measures mean execution time
over N iterations and computes speedup.
"""

from mindnlp.triton.kernels.activations import (
    triton_gelu, native_gelu,
    triton_swiglu, native_swiglu,
)

import time
import torch


def _sync(device: str):
    """Synchronize device after operations."""
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _measure(fn, args: list, warmup: int, iterations: int, device: str) -> float:
    """Measure execution time for a function."""
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    _sync(device)
    return (time.perf_counter() - start) / iterations * 1000


def run(config: dict) -> dict:
    """Run benchmark tests for activation kernels.

    Args:
        config: Pipeline configuration with optional 'device', 'benchmark' keys

    Returns:
        Dictionary containing benchmark results for all operators
    """
    device = config.get("device", "cpu")
    bench_cfg = config.get("benchmark", {})
    iterations = bench_cfg.get("iterations", 100)
    warmup = bench_cfg.get("warmup", 5)
    shapes = bench_cfg.get("shapes", [[1, 512, 4864], [72, 512, 4864]])

    results = {"gelu": [], "swiglu": []}

    for shape in shapes:
        torch.manual_seed(42)
        x = torch.randn(*shape, dtype=torch.float32, device=device).view(-1)
        gate = torch.randn(*shape, dtype=torch.float32, device=device).view(-1)
        up = torch.randn(*shape, dtype=torch.float32, device=device).view(-1)

        native_gelu_ms = _measure(native_gelu, [x], warmup, iterations, device)
        triton_gelu_ms = _measure(triton_gelu, [x], warmup, iterations, device)

        native_swiglu_ms = _measure(native_swiglu, [gate, up], warmup, iterations, device)
        triton_swiglu_ms = _measure(triton_swiglu, [gate, up], warmup, iterations, device)

        results["gelu"].append({
            "shape": shape,
            "native_ms": round(native_gelu_ms, 4),
            "triton_ms": round(triton_gelu_ms, 4),
            "speedup": round(native_gelu_ms / triton_gelu_ms, 3),
        })
        results["swiglu"].append({
            "shape": shape,
            "native_ms": round(native_swiglu_ms, 4),
            "triton_ms": round(triton_swiglu_ms, 4),
            "speedup": round(native_swiglu_ms / triton_swiglu_ms, 3),
        })

    return results