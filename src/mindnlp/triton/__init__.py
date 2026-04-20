"""triton adapter for mindspore"""
from functools import lru_cache
import mindspore
from mindtorch import ops
from mindnlp.utils import is_triton_available

if is_triton_available():
    from triton.backends.driver import DriverBase  # pylint: disable=import-error
    from triton.backends.nvidia.driver import CudaUtils, CudaLauncher  # pylint: disable=import-error
    from triton.backends.compiler import GPUTarget  # pylint: disable=import-error

    class MSDriver(DriverBase):

        def __init__(self):
            self.utils = CudaUtils()  # TODO: make static
            self.launcher_cls = CudaLauncher
            super().__init__()

        def get_current_device(self):
            return 0

        def set_current_device(self):
            pass

        @lru_cache
        def get_current_stream(self, device=None):
            return mindspore.hal.current_stream().id

        @lru_cache
        def get_device_capability(self, device=0):
            return mindspore.hal.get_device_capability(0)

        @lru_cache
        def get_current_target(self):
            device = self.get_current_device()
            capability = self.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]
            warp_size = 32
            return GPUTarget("cuda", capability, warp_size)

        def get_device_interface(self):
            return mindspore.hal

        @staticmethod
        def is_active():
            return True

        def get_benchmarker(self):
            from triton.testing import do_bench  # pylint: disable=import-error
            return do_bench

        def get_empty_cache_for_benchmark(self):
            cache_size = 256 * 1024 * 1024
            return ops.empty(int(cache_size // 4), dtype=mindspore.int32, device='GPU')


from mindnlp.triton.kernels.activations import (
    TritonGELU,
    TritonSwiGLU,
    triton_gelu,
    triton_swiglu,
    native_gelu,
    native_swiglu,
)

from mindnlp.triton.kernels.mindspore_adapter import (
    MSGELU,
    MSSwiGLU,
    TritonGELU as MSTritonGELU,
    TritonSwiGLU as MSTritonSwiGLU,
    gelu as ms_gelu,
    swiglu as ms_swiglu,
    get_ms_activation,
)

from mindnlp.triton.backends.detect import get_available_backend

__version__ = "0.1.0"

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
    "ms_gelu",
    "ms_swiglu",
    "get_ms_activation",
]