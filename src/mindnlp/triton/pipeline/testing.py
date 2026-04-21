"""
Phase 2: Numerical accuracy validation.

Compares Triton kernel outputs against native PyTorch implementations.
Pass threshold: max absolute difference < 1e-5.
"""

from mindnlp.triton.kernels.activations import (
    triton_gelu, native_gelu,
    triton_swiglu, native_swiglu,
)

import torch

PASS_THRESHOLD = 1e-5


def _test_kernel(name: str, triton_fn, native_fn, inputs: list, device: str) -> dict:
    """Test a single kernel against its native reference."""
    tensors = [
        t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32, device=device)
        for t in inputs
    ]
    with torch.no_grad():
        triton_out = triton_fn(*tensors)
        native_out = native_fn(*tensors)
    max_diff = float((triton_out - native_out).abs().max().item())
    passed = max_diff < PASS_THRESHOLD
    return {
        "max_diff": max_diff,
        "passed": passed,
        "threshold": PASS_THRESHOLD,
    }


def run(config: dict) -> dict:
    """Run numerical accuracy tests for Triton kernels.

    Args:
        config: Pipeline configuration with optional 'device' key

    Returns:
        Dictionary containing test results for all kernels
    """
    device = config.get("device", "cpu")
    torch.manual_seed(42)

    test_shape = (72, 512, 4864)
    x = torch.randn(*test_shape, dtype=torch.float32, device=device).view(-1)
    gate = torch.randn(*test_shape, dtype=torch.float32, device=device).view(-1)
    up = torch.randn(*test_shape, dtype=torch.float32, device=device).view(-1)

    results = {}
    results["gelu"] = _test_kernel("gelu", triton_gelu, native_gelu, [x], device)
    results["swiglu"] = _test_kernel("swiglu", triton_swiglu, native_swiglu, [gate, up], device)

    all_passed = all(v["passed"] for v in results.values())
    results["all_passed"] = all_passed
    return results