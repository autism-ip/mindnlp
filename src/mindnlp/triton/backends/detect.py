"""
Backend detection and selection.
"""

import os
from enum import Enum
from typing import Optional


class BackendType(Enum):
    """Available Triton backends."""
    ASCEND = "ascend"
    NVIDIA = "nvidia"
    CPU = "cpu"
    NONE = "none"


def detect_npu() -> bool:
    """Detect if Ascend NPU is available."""
    try:
        import torch
        return torch.npu.is_available()
    except Exception:
        return False


def detect_cuda() -> bool:
    """Detect if NVIDIA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_available_backend() -> BackendType:
    """Detect and return the available Triton backend.

    Returns:
        BackendType enum value indicating which backend is available
    """
    env_backend = os.environ.get("MINNLP_TRITON_BACKEND", "").lower()

    if env_backend == "ascend":
        if detect_npu():
            return BackendType.ASCEND
    elif env_backend == "nvidia":
        if detect_cuda():
            return BackendType.NVIDIA
    elif env_backend == "cpu":
        return BackendType.CPU

    if detect_npu():
        return BackendType.ASCEND
    if detect_cuda():
        return BackendType.NVIDIA

    return BackendType.NONE


def is_triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


def is_backend_available(backend: BackendType) -> bool:
    """Check if a specific backend is available.

    Args:
        backend: Backend type to check

    Returns:
        True if the backend is available, False otherwise
    """
    if backend == BackendType.NONE:
        return False

    if backend == BackendType.ASCEND:
        return detect_npu() and is_triton_available()
    elif backend == BackendType.NVIDIA:
        return detect_cuda() and is_triton_available()
    elif backend == BackendType.CPU:
        return is_triton_available()

    return False