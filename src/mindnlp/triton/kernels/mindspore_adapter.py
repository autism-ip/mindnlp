"""
MindSpore Adapter for Triton Kernels.

This module provides MindSpore-compatible interfaces to Triton kernels,
enabling seamless integration with MindSpore models.
"""

import os
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import torch
    import mindspore

TRITON_ENABLED = os.environ.get("MINNLP_TRITON", "1") == "1"


def _get_mindspore():
    """Lazy import of mindspore to avoid import-time hanging."""
    import mindspore
    return mindspore


class _TorchTensorToMindSpore:
    """Context for converting tensors during forward pass."""

    def __init__(self):
        self._temp_tensors = []

    def convert_input(self, x: "mindspore.Tensor"):
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

    def convert_output(self, torch_tensor: "torch.Tensor") -> "mindspore.Tensor":
        """Convert torch tensor back to MindSpore tensor.

        Args:
            torch_tensor: Torch tensor output

        Returns:
            MindSpore tensor
        """
        numpy_array = torch_tensor.cpu().numpy()
        self._temp_tensors.clear()
        ms = _get_mindspore()
        return ms.Tensor(numpy_array)

    def cleanup(self):
        """Cleanup temporary tensors."""
        self._temp_tensors.clear()


def _to_torch_tensor(x: "mindspore.Tensor"):
    """Convert MindSpore tensor to torch tensor."""
    import torch
    return torch.from_numpy(x.asnumpy()).contiguous()


def _to_mindspore_tensor(x):
    """Convert torch tensor to MindSpore tensor."""
    ms = _get_mindspore()
    if isinstance(x, ms.Tensor):
        return x
    return ms.Tensor(x.asnumpy())


def gelu(x) -> "mindspore.Tensor":
    """GELU activation with MindSpore tensor interface.

    Args:
        x: Input MindSpore tensor

    Returns:
        GELU activated MindSpore tensor
    """
    if TRITON_ENABLED:
        torch_x = _to_torch_tensor(x)
        from mindnlp.triton.kernels.activations import triton_gelu as _triton_gelu
        torch_out = _triton_gelu(torch_x)
        return _to_mindspore_tensor(torch_out)
    return _native_gelu_ms(x)


def swiglu(gate, up) -> "mindspore.Tensor":
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
        from mindnlp.triton.kernels.activations import triton_swiglu as _triton_swiglu
        torch_out = _triton_swiglu(torch_gate, torch_up)
        return _to_mindspore_tensor(torch_out)
    return _native_swiglu_ms(gate, up)


def _native_gelu_ms(x: "mindspore.Tensor") -> "mindspore.Tensor":
    """Native MindSpore GELU implementation."""
    ms = _get_mindspore()
    ops = ms.ops
    return x * 0.5 * (1.0 + ops.erf(x / ops.sqrt(ops.tensor(2.0, x.dtype))))


def _native_swiglu_ms(gate: "mindspore.Tensor", up: "mindspore.Tensor") -> "mindspore.Tensor":
    """Native MindSpore SwiGLU implementation."""
    ms = _get_mindspore()
    ops = ms.ops
    sigmoid_gate = gate / (1.0 + ops.exp(-gate))
    return gate * sigmoid_gate * up


class MSGELU:
    """MindSpore GELU activation using Triton kernels.

    This module provides a drop-in replacement for mindspore.nn.GELU
    that uses Triton kernels on supported hardware (Ascend NPU).

    Args:
        approximate: Not supported in current implementation
    """

    def __init__(self, approximate: str = "none"):
        self.approximate = approximate

    def construct(self, x):
        return gelu(x)


class MSSwiGLU:
    """MindSpore SwiGLU activation using Triton kernels.

    This module provides a drop-in replacement for SwiGLU activation
    commonly used in LLM models like Qwen, LLaMA, etc.

    Args:
        dim: Not used, kept for interface compatibility
    """

    def __init__(self, dim: int = -1):
        self.dim = dim

    def construct(self, gate, up):
        return swiglu(gate, up)


class TritonGELU:
    """MindSpore Cell wrapper for Triton GELU kernel.

    Provides seamless integration with MindSpore autograd.
    """

    def __init__(self):
        pass

    def construct(self, x):
        if TRITON_ENABLED:
            torch_x = _to_torch_tensor(x)
            from mindnlp.triton.kernels.activations import triton_gelu as _triton_gelu
            torch_out = _triton_gelu(torch_x)
            return _to_mindspore_tensor(torch_out)
        return _native_gelu_ms(x)


class TritonSwiGLU:
    """MindSpore Cell wrapper for Triton SwiGLU kernel.

    Provides seamless integration with MindSpore autograd.
    """

    def __init__(self):
        pass

    def construct(self, gate, up):
        if TRITON_ENABLED:
            torch_gate = _to_torch_tensor(gate)
            torch_up = _to_torch_tensor(up)
            from mindnlp.triton.kernels.activations import triton_swiglu as _triton_swiglu
            torch_out = _triton_swiglu(torch_gate, torch_up)
            return _to_mindspore_tensor(torch_out)
        return _native_swiglu_ms(gate, up)


def get_ms_activation(name: str):
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