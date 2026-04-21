"""
Triton Element-wise Activation Kernels.

Fair benchmark speedups on Ascend NPU (Triton-Ascend 3.2.0, same device comparison):
  SwiGLU: 2.58x on shape (24, 512, 4864)  - Triton faster than PyTorch Native
  GELU:   0.80x on shape (24, 512, 4864)  - PyTorch Native faster, not recommended

Note: Earlier 4.46x/2.57x numbers were NPU vs CPU comparisons (unfair).
      Fair comparison (both on NPU) shows SwiGLU benefits, GELU does not.
"""

import os
import torch
import triton
import triton.language as tl

TRITON_ENABLED = os.environ.get("MINNLP_TRITON", "1") == "1"


@triton.jit
def gelu_kernel(
    in_ptr, out_ptr, xnumel,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = x_index < xnumel
        x = tl.load(in_ptr + x_index, mask=xmask, other=0.0)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr + x_index, ret, mask=xmask)


@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr, xnumel,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = x_index < xnumel
        gate = tl.load(gate_ptr + x_index, mask=xmask, other=0.0)
        up = tl.load(up_ptr + x_index, mask=xmask, other=0.0)
        sigmoid_gate = gate / (1.0 + tl.exp(-gate))
        ret = gate * sigmoid_gate * up
        tl.store(out_ptr + x_index, ret, mask=xmask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using Triton. 0.80x speedup - PyTorch Native is faster."""
    if x.device.type == 'cpu':
        x = x.to('npu')
    out = torch.empty_like(x)
    xnumel = x.numel()
    XBLOCK = 32768
    XBLOCK_SUB = 8192
    num_blocks = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (min(num_blocks, 65535),)
    try:
        from mindnlp.triton import _patch_torch_npu_for_triton
        _patch_torch_npu_for_triton()
    except ImportError:
        pass
    gelu_kernel[grid](x, out, xnumel, XBLOCK, XBLOCK_SUB)
    return out


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU (gate * silu(gate) * up) using Triton. 2.58x faster than native on NPU."""
    assert gate.shape == up.shape, f"Shape mismatch: {gate.shape} vs {up.shape}"
    if gate.device.type == 'cpu':
        gate = gate.to('npu')
        up = up.to('npu')
    out = torch.empty_like(gate)
    xnumel = gate.numel()
    XBLOCK = 32768
    XBLOCK_SUB = 8192
    num_blocks = (xnumel + XBLOCK - 1) // XBLOCK
    grid = (min(num_blocks, 65535),)
    try:
        from mindnlp.triton import _patch_torch_npu_for_triton
        _patch_torch_npu_for_triton()
    except ImportError:
        pass
    swiglu_kernel[grid](gate, up, out, xnumel, XBLOCK, XBLOCK_SUB)
    return out


def native_gelu(x: torch.Tensor) -> torch.Tensor:
    """Native PyTorch GELU (reference baseline)."""
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))


def native_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Native PyTorch SwiGLU (reference baseline)."""
    return gate * torch.nn.functional.silu(gate) * up


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU with automatic backend selection."""
    if TRITON_ENABLED:
        return triton_gelu(x)
    return native_gelu(x)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU with automatic backend selection."""
    if TRITON_ENABLED:
        return triton_swiglu(gate, up)
    return native_swiglu(gate, up)


class TritonGELU(torch.nn.Module):
    """Drop-in nn.Module replacement for GELU using Triton kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TRITON_ENABLED:
            return triton_gelu(x)
        return native_gelu(x)


class TritonSwiGLU(torch.nn.Module):
    """Drop-in nn.Module replacement for SwiGLU using Triton kernel."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if TRITON_ENABLED:
            return triton_swiglu(gate, up)
        return native_swiglu(gate, up)