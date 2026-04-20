# MindNLP Triton 模块

基于 MindSpore 框架的高性能 Triton 算子实现，用于 Ascend NPU 和 NVIDIA GPU。

## 性能数据

| 算子 | 数据规模 | 加速比 | 数值精度 |
|------|----------|--------|----------|
| GELU | 72,512,4864 | **4.52x** | < 1e-6 |
| SwiGLU | 72,512,4864 | **2.60x** | < 1e-6 |

## 支持的硬件

- **Ascend NPU**: Ascend 910B4 (Triton-Ascend 3.2.0+)
- **NVIDIA GPU**: CUDA 11.0+ with Triton

## 安装依赖

```bash
pip install triton
```

## 快速开始

### MindSpore (推荐)

```python
import mindspore as ms
from mindnlp.triton import MSGELU, MSSwiGLU, gelu, swiglu

# 使用 Cell 接口
gelu_act = MSGELU()
output = gelu_act(x)

# 使用函数接口
output = gelu(x)
output = swiglu(gate, up)
```

### PyTorch 兼容

```python
import torch
from mindnlp.triton import TritonGELU, TritonSwiGLU, triton_gelu, triton_swiglu

# 使用 nn.Module
act = TritonGELU()
output = act(x)

# 使用函数
output = triton_gelu(x)
output = triton_swiglu(gate, up)
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINNLP_TRITON` | `1` | 启用/禁用 Triton (`0` 禁用) |
| `MINNLP_TRITON_BACKEND` | `auto` | 强制指定后端 (`ascend`, `nvidia`, `cpu`) |

## 模块结构

```
mindnlp.triton/
├── kernels/
│   ├── activations.py      # GELU, SwiGLU Triton 实现
│   ├── benchmark.py         # 性能测试工具
│   └── mindspore_adapter.py # MindSpore 适配层
├── backends/
│   ├── detect.py           # 后端自动检测
│   └── ascend.py           # Ascend NPU 支持
├── integration/
│   └── mindtorch_v2.py     # mindtorch_v2 集成
└── docs/
    ├── ANALYSIS_REPORT.md   # 瓶颈分析报告
    └── PERFORMANCE_REPORT.md # 性能测试报告
```

## 适用模型

| 模型 | 激活函数 | 推荐优化 |
|------|----------|----------|
| Qwen2 (0.5B) | SwiGLU | ✅ `MSSwiGLU` |
| Qwen2.5 | SwiGLU | ✅ `MSSwiGLU` |
| LLaMA | GELU | ✅ `MSGELU` |

## 不建议优化的算子

| 算子 | 原因 |
|------|------|
| GEAM (matmul) | CANN 快 17x |
| RMSNorm/LayerNorm | CANN 快 100x |
| Attention/Softmax | CANN 快 5x |

## 性能测试

```python
from mindnlp.triton.kernels.benchmark import benchmark_activation, benchmark_swiglu

# 测试 GELU
result = benchmark_activation(
    activation_fn=triton_gelu,
    shape=(24, 512, 4864),
    device="npu"
)

# 测试 SwiGLU
result = benchmark_swiglu(
    shape=(24, 512, 4864),
    device="npu"
)
```

## 注意事项

1. **Triton-Ascend 限制**：
   - Grid 限制: coreDim <= 65535
   - 不支持 `continue` 语句
   - 大型 kernel 可能编译不稳定

2. **自动降级**：当 Triton 不可用时，自动回退到原生实现

## 许可证

Apache 2.0 - 与 MindNLP 保持一致