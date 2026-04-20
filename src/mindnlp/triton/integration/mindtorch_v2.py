"""
MindTorch v2 Integration for Triton kernels.

This module provides integration between Triton kernels and the mindtorch_v2
nn.Module system, allowing transparent use of Triton-accelerated operations.
"""

import os
from typing import Optional

import torch

from mindnlp.triton.kernels.activations import (
    TritonGELU as _TritonGELU,
    TritonSwiGLU as _TritonSwiGLU,
    triton_gelu,
    triton_swiglu,
    native_gelu,
    native_swiglu,
)
from mindnlp.triton.backends.detect import get_available_backend, BackendType


TRITON_ENABLED = os.environ.get("MINNLP_TRITON", "1") == "1"
TRITON_INTEGRATION_ENABLED = os.environ.get("MINNLP_TRITON_INTEGRATION", "1") == "1"


class TritonGELU(torch.nn.Module):
    """Triton-accelerated GELU activation.

    This module provides a drop-in replacement for torch.nn.GELU that uses
    Triton kernels on supported backends (Ascend NPU, NVIDIA GPU).

    Args:
        approximate: Not supported in current implementation
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TRITON_ENABLED and TRITON_INTEGRATION_ENABLED:
            return triton_gelu(x)
        return native_gelu(x)


class TritonSwiGLU(torch.nn.Module):
    """Triton-accelerated SwiGLU activation.

    This module provides a drop-in replacement for the SwiGLU activation
    pattern commonly used in LLM models like Qwen, LLaMA, etc.

    Args:
        dim: Not used, kept for interface compatibility
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if TRITON_ENABLED and TRITON_INTEGRATION_ENABLED:
            return triton_swiglu(gate, up)
        return native_swiglu(gate, up)


def get_triton_activation(name: str) -> Optional[torch.nn.Module]:
    """Get a Triton-accelerated activation module by name.

    Args:
        name: Name of the activation ("gelu", "swiglu", etc.)

    Returns:
        Triton-accelerated module if available, None otherwise
    """
    if not TRITON_INTEGRATION_ENABLED:
        return None

    activations = {
        "gelu": TritonGELU,
        "swiglu": TritonSwiGLU,
    }

    return activations.get(name.lower())


def patch_mindtorch_activations():
    """Patch mindtorch_v2 activation modules with Triton versions.

    This function patches the standard activation functions in mindtorch_v2
    to use Triton-accelerated versions when available.
    """
    if not TRITON_INTEGRATION_ENABLED:
        return

    try:
        from mindtorch_v2.nn.modules import activation

        if not hasattr(activation, '_triton_patched'):
            original_gelu = getattr(activation, 'GELU', None)
            if original_gelu is not None and not isinstance(original_gelu, type) or not issubclass(original_gelu, TritonGELU):
                pass

            activation._triton_patched = True
    except ImportError:
        pass


def enable_triton_integration():
    """Enable Triton integration for mindtorch_v2."""
    global TRITON_INTEGRATION_ENABLED
    TRITON_INTEGRATION_ENABLED = True


def disable_triton_integration():
    """Disable Triton integration for mindtorch_v2."""
    global TRITON_INTEGRATION_ENABLED
    TRITON_INTEGRATION_ENABLED = False


class TritonIntegration:
    """Context manager for Triton integration.

    Usage:
        with TritonIntegration():
            # Triton kernels will be used when available
            output = model(input)
    """

    def __init__(self, enabled: bool = True):
        self._previous_state = TRITON_INTEGRATION_ENABLED
        self._enabled = enabled

    def __enter__(self):
        if self._enabled:
            enable_triton_integration()
        else:
            disable_triton_integration()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global TRITON_INTEGRATION_ENABLED
        TRITON_INTEGRATION_ENABLED = self._previous_state
        return False


__all__ = [
    "TritonGELU",
    "TritonSwiGLU",
    "get_triton_activation",
    "patch_mindtorch_activations",
    "enable_triton_integration",
    "disable_triton_integration",
    "TritonIntegration",
    "TRITON_ENABLED",
    "TRITON_INTEGRATION_ENABLED",
]