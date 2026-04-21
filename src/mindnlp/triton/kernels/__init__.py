"""
Triton Kernel Implementations.
"""

from mindnlp.triton.kernels.activations import (
    TritonGELU,
    TritonSwiGLU,
    triton_gelu,
    triton_swiglu,
    native_gelu,
    native_swiglu,
)

from mindnlp.triton.kernels.benchmark import (
    benchmark_activation,
    benchmark_swiglu,
)

from mindnlp.triton.kernels.mindspore_adapter import (
    MSGELU,
    MSSwiGLU,
    TritonGELU as MSTritonGELU,
    TritonSwiGLU as MSTritonSwiGLU,
    gelu,
    swiglu,
    get_ms_activation,
    is_triton_available,
    TRITON_ENABLED,
)

__all__ = [
    "TritonGELU",
    "TritonSwiGLU",
    "triton_gelu",
    "triton_swiglu",
    "native_gelu",
    "native_swiglu",
    "benchmark_activation",
    "benchmark_swiglu",
    "MSGELU",
    "MSSwiGLU",
    "MSTritonGELU",
    "MSTritonSwiGLU",
    "gelu",
    "swiglu",
    "get_ms_activation",
    "is_triton_available",
    "TRITON_ENABLED",
]