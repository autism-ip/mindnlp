"""
MindTorch v2 Integration for Triton kernels.
"""

from mindnlp.triton.integration.mindtorch_v2 import (
    TritonGELU,
    TritonSwiGLU,
    get_triton_activation,
    patch_mindtorch_activations,
    enable_triton_integration,
    disable_triton_integration,
    TritonIntegration,
    TRITON_ENABLED,
    TRITON_INTEGRATION_ENABLED,
)

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