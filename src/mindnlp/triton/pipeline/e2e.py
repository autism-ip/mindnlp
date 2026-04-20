"""
Phase 4: End-to-End MLP performance validation.

Tests full MLP layer: matmul (CANN) + Triton activation.
Compares against native silu baseline.

Config shape: [batch, seq_len, hidden_size, intermediate_size]
"""

from mindnlp.triton.kernels.activations import triton_gelu, triton_swiglu

import time
import torch
import torch.nn.functional as F


def _sync(device: str):
    """Synchronize device after operations."""
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _mlp_native_silu(x, gate_w, up_w, down_w):
    """MLP with native silu activation."""
    gate_out = torch.matmul(x, gate_w.t())
    up_out = torch.matmul(x, up_w.t())
    activated = F.silu(gate_out) * up_out
    return torch.matmul(activated, down_w.t())


def _mlp_triton_gelu(x, gate_w, up_w, down_w):
    """MLP with Triton GELU activation."""
    gate_out = torch.matmul(x, gate_w.t())
    up_out = torch.matmul(x, up_w.t())
    activated = triton_gelu(gate_out) * up_out
    return torch.matmul(activated, down_w.t())


def _mlp_triton_swiglu(x, gate_w, up_w, down_w):
    """MLP with Triton SwiGLU activation."""
    gate_out = torch.matmul(x, gate_w.t())
    up_out = torch.matmul(x, up_w.t())
    activated = triton_swiglu(gate_out, up_out)
    return torch.matmul(activated, down_w.t())


def _measure_mlp(fn, args, warmup, iterations, device):
    """Measure MLP layer execution time."""
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    _sync(device)
    return (time.perf_counter() - start) / iterations * 1000


def run(config: dict) -> dict:
    """Run end-to-end MLP benchmarks.

    Args:
        config: Pipeline configuration with optional 'device', 'e2e' keys

    Returns:
        Dictionary containing MLP benchmark results
    """
    device = config.get("device", "cpu")
    e2e_cfg = config.get("e2e", {})
    iterations = e2e_cfg.get("iterations", 100)
    warmup = e2e_cfg.get("warmup", 5)
    configs = e2e_cfg.get("configs", [
        [1, 512, 896, 4864],
        [24, 512, 896, 4864],
    ])

    results = {"mlp_with_gelu": [], "mlp_with_swiglu": []}

    for cfg in configs:
        batch, seq_len, hidden_size, intermediate_size = cfg
        torch.manual_seed(42)
        x = torch.randn(batch * seq_len, hidden_size, dtype=torch.float32, device=device)
        gate_w = torch.randn(intermediate_size, hidden_size, dtype=torch.float32, device=device)
        up_w = torch.randn(intermediate_size, hidden_size, dtype=torch.float32, device=device)
        down_w = torch.randn(hidden_size, intermediate_size, dtype=torch.float32, device=device)
        args = (x, gate_w, up_w, down_w)

        native_ms = _measure_mlp(_mlp_native_silu, args, warmup, iterations, device)
        gelu_ms = _measure_mlp(_mlp_triton_gelu, args, warmup, iterations, device)
        swiglu_ms = _measure_mlp(_mlp_triton_swiglu, args, warmup, iterations, device)

        results["mlp_with_gelu"].append({
            "config": cfg,
            "native_ms": round(native_ms, 4),
            "triton_ms": round(gelu_ms, 4),
            "speedup": round(native_ms / gelu_ms, 3),
        })
        results["mlp_with_swiglu"].append({
            "config": cfg,
            "native_ms": round(native_ms, 4),
            "triton_ms": round(swiglu_ms, 4),
            "speedup": round(native_ms / swiglu_ms, 3),
        })

    return results