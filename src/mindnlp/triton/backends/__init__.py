"""
Triton Backend Implementations.
"""

from mindnlp.triton.backends.detect import (
    get_available_backend,
    is_triton_available,
    BackendType,
)
from mindnlp.triton.backends.ascend import (
    get_ascend_backend,
    is_ascend_available,
)

__all__ = [
    "BackendType",
    "get_available_backend",
    "is_triton_available",
    "get_ascend_backend",
    "is_ascend_available",
]