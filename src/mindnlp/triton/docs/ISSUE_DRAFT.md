# [Feature] Qwen 模型 Triton SwiGLU 激活函数优化 (2.58x 加速)

## 问题描述

基于 MindSpore 框架，使用 Triton 技术栈为 Qwen 模型开发高性能替换算子，提升大模型在 Ascend NPU 上的推理效率。

## 背景

当前 MindSpore 需要通过 **Triton** 技术栈进行高性能算子替换，以提升大模型在特定硬件上的推理与训练效率。

## 功能描述

### 实现内容

在 `mindnlp/triton` 模块中集成了 Triton 激活函数优化：

1. **Triton SwiGLU** - Qwen2/Qwen2.5 等模型使用的激活函数
   - 位置: `mindnlp/triton/kernels/activations.py`
   - 加速比: **2.58x** (相比 PyTorch Native on NPU)

2. **Triton GELU** - LLaMA 等模型使用的激活函数
   - 位置: `mindnlp/triton/kernels/activations.py`
   - 加速比: 0.80x (PyTorch Native 更快，不推荐)

### 性能数据 (Ascend NPU, 公平对比)

| 算子 | PyTorch Native | Triton | 加速比 |
|------|----------------|--------|--------|
| **SwiGLU** | 3.06ms | 1.19ms | **2.58x** |
| GELU | 0.75ms | 0.94ms | 0.80x |

**测试配置**: (24, 512, 4864) - Qwen2-0.5B 24层批量
**测试条件**: Triton 和 PyTorch Native 都在 NPU 上运行

### 瓶颈分析

**Qwen2.5-0.5B 模型 Profiling 结果**：

| 模块 | 时间占比 | 说明 |
|------|----------|------|
| MLP | **28.3%** | 包含 matmul + SwiGLU 激活 |
| Attention | 31.9% | QKV projection + attention |
| LayerNorm | 39.5% | RMSNorm |

**关键发现**：
- MLP 层中 SwiGLU 激活函数是可优化的目标
- GEAM (matmul) 占 96% 总时间，但 CANN 已极度优化 (比 Triton 快 17x)

## API 使用示例

### PyTorch 兼容方式

```python
import torch
from mindnlp.triton import triton_gelu, triton_swiglu

# SwiGLU 激活 (推荐用于 Qwen2/Qwen2.5)
gate = torch.randn(24, 512, 4864, device='npu')
up = torch.randn(24, 512, 4864, device='npu')
output = triton_swiglu(gate, up)

# GELU 激活
x = torch.randn(24, 512, 4864, device='npu')
output = triton_gelu(x)
```

### MindSpore 方式 (推荐)

```python
import mindspore as ms
from mindnlp.triton import MSGELU, MSSwiGLU, gelu, swiglu

# Cell 方式
gelu_act = MSGELU()
output = gelu_act(x)

swiglu_act = MSSwiGLU()
output = swiglu_act(gate, up)

# 函数方式
output = gelu(x)
output = swiglu(gate, up)
```

## 适用模型

| 模型 | 激活函数 | 推荐优化 | 加速比 |
|------|----------|----------|--------|
| Qwen2 (0.5B) | SwiGLU | `MSSwiGLU` | **2.58x** |
| Qwen2.5 | SwiGLU | `MSSwiGLU` | **2.58x** |
| LLaMA | GELU | ❌ 不推荐 | 0.80x |

## 模块结构

```
mindnlp/triton/
├── __init__.py                 # 主入口，导出所有公开 API
├── kernels/
│   ├── activations.py          # Triton GELU/SwiGLU 实现
│   ├── benchmark.py           # 性能测试工具
│   └── mindspore_adapter.py   # MindSpore 适配层 (MSGELU, MSSwiGLU)
├── backends/                   # 后端检测
├── pipeline/                  # 优化管线
└── docs/
    ├── ANALYSIS_REPORT.md      # 瓶颈分析报告
    └── PERFORMANCE_REPORT.md  # 性能测试报告
```

## 环境要求

- **硬件**: Atlas 800I A2 / 800T A2
- **CANN**: ≥ 8.1.RC1
- **MindSpore**: ≥ 2.7.0
- **Triton**: ≥ 3.2.0 (Ascend 适配版)

## 建议

1. **对于使用 SwiGLU 的模型**（如 Qwen2、Qwen2.5）：
   - 使用 `swiglu()` 或 `MSSwiGLU()` 替代原生实现
   - 预期激活函数加速 **2.58x**

2. **不要尝试用 Triton 替代 CANN matmul**：
   - 性能差距太大 (17x)
   - CANN 已极度优化

## 相关文档

- [详细性能报告](./docs/PERFORMANCE_REPORT.md)
- [瓶颈分析报告](./docs/ANALYSIS_REPORT.md)
