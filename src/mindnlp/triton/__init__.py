"""triton adapter for mindspore"""
import os as _os
import sys as _sys
from functools import lru_cache

# Set TRITON_BACKEND to mindspore to avoid torch_npu._C dependency issues
# when MindTorchFinder proxies torch_npu imports
if _os.environ.get('TRITON_BACKEND') not in ('torch_npu', 'mindspore'):
    _os.environ['TRITON_BACKEND'] = 'mindspore'


def _get_mindspore():
    """Lazy import mindspore to avoid import-time hanging."""
    import mindspore
    return mindspore


def _get_mindtorch_ops():
    """Lazy import mindtorch.ops."""
    from mindtorch import ops
    return ops


def _register_ms_driver():
    """Register MSDriver for Triton if triton is available."""
    try:
        from mindnlp.utils import is_triton_available
        if not is_triton_available():
            return
    except ImportError:
        return

    try:
        from triton.backends.driver import DriverBase
        from triton.backends.nvidia.driver import CudaUtils, CudaLauncher
        from triton.backends.compiler import GPUTarget

        class MSDriver(DriverBase):

            def __init__(self):
                self.utils = CudaUtils()
                self.launcher_cls = CudaLauncher
                super().__init__()

            def get_current_device(self):
                return 0

            def set_current_device(self):
                pass

            @lru_cache
            def get_current_stream(self, device=None):
                ms = _get_mindspore()
                return ms.hal.current_stream().id

            @lru_cache
            def get_device_capability(self, device=0):
                ms = _get_mindspore()
                return ms.hal.get_device_capability(0)

            @lru_cache
            def get_current_target(self):
                device = self.get_current_device()
                capability = self.get_device_capability(device)
                capability = capability[0] * 10 + capability[1]
                warp_size = 32
                return GPUTarget("cuda", capability, warp_size)

            def get_device_interface(self):
                ms = _get_mindspore()
                return ms.hal

            @staticmethod
            def is_active():
                return True

            def get_benchmarker(self):
                from triton.testing import do_bench
                return do_bench

            def get_empty_cache_for_benchmark(self):
                cache_size = 256 * 1024 * 1024
                ops = _get_mindtorch_ops()
                ms = _get_mindspore()
                return ops.empty(int(cache_size // 4), dtype=ms.int32, device='GPU')

    except (ImportError, AttributeError):
        pass


_register_ms_driver()

__all__ = [
    "TritonGELU",
    "TritonSwiGLU",
    "triton_gelu",
    "triton_swiglu",
    "native_gelu",
    "native_swiglu",
    "get_available_backend",
    "MSGELU",
    "MSSwiGLU",
    "MSTritonGELU",
    "MSTritonSwiGLU",
    "gelu",
    "swiglu",
    "get_ms_activation",
]

def __getattr__(name):
    if name in ("TritonGELU", "TritonSwiGLU", "triton_gelu", "triton_swiglu", "native_gelu", "native_swiglu"):
        from mindnlp.triton.kernels.activations import (
            TritonGELU, TritonSwiGLU, triton_gelu, triton_swiglu, native_gelu, native_swiglu,
        )
        globals().update(locals())
        return locals()[name]
    if name in ("MSGELU", "MSSwiGLU", "MSTritonGELU", "MSTritonSwiGLU", "gelu", "swiglu", "get_ms_activation"):
        from mindnlp.triton.kernels.mindspore_adapter import (
            MSGELU, MSSwiGLU, TritonGELU as MSTritonGELU, TritonSwiGLU as MSTritonSwiGLU,
            gelu, swiglu, get_ms_activation,
        )
        globals().update(locals())
        return locals()[name]
    if name == "get_available_backend":
        from mindnlp.triton.backends.detect import get_available_backend
        globals()["get_available_backend"] = get_available_backend
        return get_available_backend
    raise AttributeError(f"module 'mindnlp.triton' has no attribute '{name}'")