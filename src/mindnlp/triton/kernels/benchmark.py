"""
Benchmark utilities for Triton kernels.

公平性能对比测试：Triton vs PyTorch Native (在同一设备上)
"""

import torch
import time
from typing import Callable, Dict, Any, Tuple


def benchmark_function(
    func: Callable,
    *args,
    warmup: int = 10,
    runs: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Benchmark a function with given inputs.

    Args:
        func: Function to benchmark
        *args: Positional arguments to pass to func
        warmup: Number of warmup runs
        runs: Number of benchmark runs
        **kwargs: Keyword arguments to pass to func

    Returns:
        Dictionary with timing statistics
    """
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if torch.is_tensor(result):
            if result.is_npu:
                torch.npu.synchronize()
            elif result.is_cuda:
                torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.is_tensor(result):
            if result.is_npu:
                torch.npu.synchronize()
            elif result.is_cuda:
                torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "runs": runs,
    }


def benchmark_activation(
    activation_fn: Callable,
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "npu",
    **kwargs
) -> Dict[str, Any]:
    """Benchmark an activation function.

    Args:
        activation_fn: Activation function to benchmark
        shape: Input shape
        dtype: Data type
        device: Device to run on
        **kwargs: Additional arguments for activation_fn

    Returns:
        Dictionary with benchmark results
    """
    x = torch.randn(shape, dtype=dtype, device=device)
    result = benchmark_function(activation_fn, x, **kwargs)
    return {
        "shape": shape,
        "dtype": str(dtype),
        "device": device,
        "mean_ms": result["mean"] * 1000,
        "min_ms": result["min"] * 1000,
        "max_ms": result["max"] * 1000,
    }


def native_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch native GELU as reference baseline."""
    return torch.nn.functional.gelu(x)


def native_swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch native SwiGLU as reference baseline."""
    return gate * torch.nn.functional.silu(gate) * up


def benchmark_swiglu(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "npu",
) -> Dict[str, Any]:
    """Benchmark SwiGLU activation with fair comparison (same device).

    Compares Triton swiglu against PyTorch native swiglu on the same device.

    Args:
        shape: Input shape (applied to both gate and up)
        dtype: Data type
        device: Device to run on

    Returns:
        Dictionary with benchmark results for both native and triton
    """
    from mindnlp.triton.kernels.activations import triton_swiglu

    gate = torch.randn(shape, dtype=dtype, device=device)
    up = torch.randn(shape, dtype=dtype, device=device)

    native_result = benchmark_function(native_swiglu_ref, gate, up)
    triton_result = benchmark_function(triton_swiglu, gate, up)

    speedup = native_result["mean"] / triton_result["mean"]

    return {
        "shape": shape,
        "dtype": str(dtype),
        "device": device,
        "native_ms": native_result["mean"] * 1000,
        "triton_ms": triton_result["mean"] * 1000,
        "speedup": speedup,
    }


def compare_activations(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "npu",
) -> Dict[str, Any]:
    """Compare native vs Triton activation functions with fair benchmark.

    Both native and Triton versions run on the same device for fair comparison.
    Uses torch.nn.functional.gelu/silu as the reference baseline.

    Args:
        shape: Input shape
        dtype: Data type
        device: Device to run on

    Returns:
        Dictionary with comparison results for GELU and SwiGLU
    """
    from mindnlp.triton.kernels.activations import triton_gelu, triton_swiglu

    results = {}

    x = torch.randn(shape, dtype=dtype, device=device)

    native_gelu_result = benchmark_function(native_gelu_ref, x)
    triton_gelu_result = benchmark_function(triton_gelu, x)
    results["gelu"] = {
        "shape": shape,
        "native_ms": native_gelu_result["mean"] * 1000,
        "triton_ms": triton_gelu_result["mean"] * 1000,
        "speedup": native_gelu_result["mean"] / triton_gelu_result["mean"],
    }

    gate = torch.randn(shape, dtype=dtype, device=device)
    up = torch.randn(shape, dtype=dtype, device=device)
    native_swiglu_result = benchmark_function(native_swiglu_ref, gate, up)
    triton_swiglu_result = benchmark_function(triton_swiglu, gate, up)
    results["swiglu"] = {
        "shape": shape,
        "native_ms": native_swiglu_result["mean"] * 1000,
        "triton_ms": triton_swiglu_result["mean"] * 1000,
        "speedup": native_swiglu_result["mean"] / triton_swiglu_result["mean"],
    }

    return results


def run_fair_benchmark(
    shape: Tuple[int, ...] = (24, 512, 4864),
    device: str = "npu",
    warmup: int = 10,
    runs: int = 100,
) -> Dict[str, Any]:
    """Run fair benchmark comparing Triton vs Native on NPU.

    This is the recommended way to benchmark as it ensures both
    implementations run on the same device under identical conditions.

    Args:
        shape: Input shape for the benchmark
        device: Device to run on (npu or cuda)
        warmup: Number of warmup iterations
        runs: Number of benchmark iterations

    Returns:
        Dictionary with all benchmark results and analysis
    """
    results = compare_activations(shape=shape, dtype=torch.float32, device=device)

    gelu_speedup = results["gelu"]["speedup"]
    swiglu_speedup = results["swiglu"]["speedup"]

    analysis = {
        "configuration": {
            "shape": shape,
            "device": device,
            "dtype": "float32",
            "warmup": warmup,
            "runs": runs,
        },
        "results": results,
        "summary": {
            "gelu_speedup": f"{gelu_speedup:.2f}x",
            "swiglu_speedup": f"{swiglu_speedup:.2f}x",
            "recommendation": "Triton recommended" if gelu_speedup > 1 and swiglu_speedup > 1 else "Check implementation",
        },
    }

    return analysis