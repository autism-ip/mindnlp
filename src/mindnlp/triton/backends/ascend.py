"""
Ascend NPU backend implementation.
"""

from typing import Optional, List
import torch

from mindnlp.triton.backends.detect import BackendType, is_triton_available


class AscendBackend:
    """Ascend NPU backend for Triton kernels."""

    def __init__(self):
        self._device = "npu"
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the Ascend backend.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return True

        try:
            if not torch.npu.is_available():
                return False

            import triton
            self._initialized = True
            return True
        except Exception:
            return False

    @property
    def device(self) -> str:
        """Return the device string for this backend."""
        return self._device

    @property
    def device_count(self) -> int:
        """Return the number of available devices."""
        return torch.npu.device_count()

    def synchronize(self, device_id: Optional[int] = None):
        """Synchronize the device.

        Args:
            device_id: Optional device ID to synchronize
        """
        if device_id is not None:
            with torch.npu.device(f"npu:{device_id}"):
                torch.npu.synchronize()
        else:
            torch.npu.synchronize()

    def set_device(self, device_id: int):
        """Set the current device.

        Args:
            device_id: Device ID to set
        """
        torch.npu.set_device(device_id)


_ascend_backend_instance: Optional[AscendBackend] = None


def get_ascend_backend() -> AscendBackend:
    """Get the global Ascend backend instance.

    Returns:
        AscendBackend instance
    """
    global _ascend_backend_instance
    if _ascend_backend_instance is None:
        _ascend_backend_instance = AscendBackend()
    return _ascend_backend_instance


def is_ascend_available() -> bool:
    """Check if Ascend backend is available.

    Returns:
        True if Ascend NPU is available, False otherwise
    """
    return torch.npu.is_available() and is_triton_available()


def is_triton_available_for_ascend() -> bool:
    """Check if Triton is available and can target Ascend.

    Returns:
        True if Triton can target Ascend, False otherwise
    """
    try:
        import triton
        return hasattr(triton, 'runtime')
    except Exception:
        return False