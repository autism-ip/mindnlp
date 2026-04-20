"""
MindSpore Adapter for Triton Kernels.

This module provides MindSpore-compatible interfaces to Triton kernels,
enabling seamless integration with MindSpore models.
"""

import os
from typing import Optional, Tuple

import mindspore
from mindspore import nn, ops

from mindnlp.triton.kernels.activations import (
    TritonGELU as _TritonGELU,
    TritonSwiGLU as _TritonSwiGLU,
    triton_gelu as _triton_gelu,
    triton_swiglu as _triton_swiglu,
    native_gelu as _native_gelu,
    native_swiglu as _native_swiglu,
)

TRITON_ENABLED = os.environ.get("MINNLP_TRITON", "1") == "1"


class _TorchTensorToMindSpore:
    """Context for converting tensors during forward pass."""

    def __init__(self):
        self._temp_tensors = []

    def convert_input(self, x: mindspore.Tensor) -> Tuple[mindspore.Tensor, any]:
        """Convert MindSpore tensor to torch tensor for Triton kernel.

        Args:
            x: MindSpore tensor

        Returns:
            Tuple of (converted tensor, cleanup handle)
        """
        import torch
        torch_tensor = torch.from_numpy(x.asnumpy()).contiguous()
        self._temp_tensors.append(torch_tensor)
        return torch_tensor, None

    def convert_output(self, torch_tensor: torch.Tensor) -> mindspore.Tensor:
        """Convert torch tensor back to MindSpore tensor.

        Args:
            torch_tensor: Torch tensor output

        Returns:
            MindSpore tensor
        """
        numpy_array = torch_tensor.cpu().numpy()
        self._temp_tensors.clear()
        return mindspore.Tensor(numpy_array)

    def cleanup(self):
        """Cleanup temporary tensors."""
        self._temp_tensors.clear()


def _to_torch_tensor(x: mindspore.Tensor):
    """Convert MindSpore tensor to torch tensor."""
    import torch
    return torch.from_numpy(x.asnumpy()).contiguous()


def _to_mindspore_tensor(x) -> mindspore.Tensor:
    """Convert torch tensor to MindSpore tensor."""
    if isinstance(x, mindspore.Tensor):
        return x
    return mindspore.Tensor(x.asnumpy())


def gelu(x: mindspore.Tensor) -> mindspore.Tensor:
    """GELU activation with MindSpore tensor interface.

    Args:
        x: Input MindSpore tensor

    Returns:
        GELU activated MindSpore tensor
    """
    if TRITON_ENABLED:
        torch_x = _to_torch_tensor(x)
        torch_out = _triton_gelu(torch_x)
        return _to_mindspore_tensor(torch_out)
    return _native_gelu_ms(x)


def swiglu(gate: mindspore.Tensor, up: mindspore.Tensor) -> mindspore.Tensor:
    """SwiGLU activation with MindSpore tensor interface.

    Args:
        gate: Gate tensor
        up: Up tensor

    Returns:
        SwiGLU activated MindSpore tensor
    """
    if TRITON_ENABLED:
        torch_gate = _to_torch_tensor(gate)
        torch_up = _to_torch_tensor(up)
        torch_out = _triton_swiglu(torch_gate, torch_up)
        return _to_mindspore_tensor(torch_out)
    return _native_swiglu_ms(gate, up)


def _native_gelu_ms(x: mindspore.Tensor) -> mindspore.Tensor:
    """Native MindSpore GELU implementation."""
    return x * 0.5 * (1.0 + ops.erf(x / ops.sqrt(ops.tensor(2.0, x.dtype))))


def _native_swiglu_ms(gate: mindspore.Tensor, up: mindspore.Tensor) -> mindspore.Tensor:
    """Native MindSpore SwiGLU implementation."""
    sigmoid_gate = gate / (1.0 + ops.exp(-gate))
    return gate * sigmoid_gate * up


class MSGELU(nn.Cell):
    """MindSpore GELU activation using Triton kernels.

    This module provides a drop-in replacement for mindspore.nn.GELU
    that uses Triton kernels on supported hardware (Ascend NPU).

    Args:
        approximate: Not supported in current implementation
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        return gelu(x)


class MSSwiGLU(nn.Cell):
    """MindSpore SwiGLU activation using Triton kernels.

    This module provides a drop-in replacement for SwiGLU activation
    commonly used in LLM models like Qwen, LLaMA, etc.

    Args:
        dim: Not used, kept for interface compatibility
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def construct(self, gate: mindspore.Tensor, up: mindspore.Tensor) -> mindspore.Tensor:
        return swiglu(gate, up)


class TritonGELU(nn.Cell):
    """MindSpore Cell wrapper for Triton GELU kernel.

    Provides seamless integration with MindSpore autograd.
    """

    def __init__(self):
        super().__init__()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        if TRITON_ENABLED:
            torch_x = _to_torch_tensor(x)
            torch_out = _triton_gelu(torch_x)
            return _to_mindspore_tensor(torch_out)
        return _native_gelu_ms(x)


class TritonSwiGLU(nn.Cell):
    """MindSpore Cell wrapper for Triton SwiGLU kernel.

    Provides seamless integration with MindSpore autograd.
    """

    def __init__(self):
        super().__init__()

    def construct(self, gate: mindspore.Tensor, up: mindspore.Tensor) -> mindspore.Tensor:
        if TRITON_ENABLED:
            torch_gate = _to_torch_tensor(gate)
            torch_up = _to_torch_tensor(up)
            torch_out = _triton_swiglu(torch_gate, torch_up)
            return _to_mindspore_tensor(torch_out)
        return _native_swiglu_ms(gate, up)


def get_ms_activation(name: str) -> Optional[nn.Cell]:
    """Get MindSpore activation module by name.

    Args:
        name: Name of the activation ("gelu", "swiglu", etc.)

    Returns:
        MindSpore Cell if available, None otherwise
    """
    activations = {
        "gelu": MSGELU,
        "swiglu": MSSwiGLU,
        "triton_gelu": TritonGELU,
        "triton_swiglu": TritonSwiGLU,
    }
    return activations.get(name.lower())


def is_triton_available() -> bool:
    """Check if Triton is available.

    Returns:
        True if Triton is installed and can be used
    """
    try:
        import triton
        return True
    except ImportError:
        return False


__all__ = [
    "MSGELU",
    "MSSwiGLU",
    "TritonGELU",
    "TritonSwiGLU",
    "gelu",
    "swiglu",
    "get_ms_activation",
    "is_triton_available",
    "TRITON_ENABLED",
]