# MindNLP Triton 模块

基于 MindSpore 框架的高性能 Triton 算子实现，用于 Ascend NPU 和 NVIDIA GPU。

## 性能数据

**实测结果 (Ascend NPU)**：

| 算子 | 数据规模 | Native | Triton | 加速比 |
|------|----------|--------|--------|--------|
| GELU | 24,512,4864 | 41.96ms | 1.01ms | **41.68x** |
| SwiGLU | 24,512,4864 | 9.00ms | 1.26ms | **7.16x** |

数值精度：< 1e-6

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

## 优化管线 (Pipeline)

完整的 Qwen 模型 Triton 优化管线，包含 5 个阶段：

```python
from mindnlp.triton.pipeline import run_pipeline, run_all

# 运行指定阶段
config = {
    'model': 'qwen2.5-0.5b',
    'device': 'npu',
    'benchmark': {
        'iterations': 100,
        'warmup': 5,
        'shapes': [[24, 512, 4864]]
    },
    'e2e': {
        'iterations': 100,
        'warmup': 5,
        'configs': [[24, 512, 896, 4864]]
    }
}
results = run_pipeline(config, ['profiling', 'benchmark', 'e2e', 'report'])

# 运行全部阶段
results = run_all(config)
```

### 管线阶段

| 阶段 | 说明 |
|------|------|
| `profiling` | 分析 Qwen 模型性能数据，识别瓶颈算子 |
| `test` | 验证 Triton kernel 数值精度 (阈值 < 1e-5) |
| `benchmark` | 单算子性能对比 (Triton vs Native) |
| `e2e` | 端到端 MLP 性能验证 |
| `report` | 生成汇总报告和优化建议 |

### CLI 使用

```bash
# 运行全部阶段
python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --phase all

# 运行指定阶段
python -m mindnlp.triton.pipeline --model qwen2-0.5b --phase profiling,benchmark

# 输出到文件
python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --phase all --output results.json
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINNLP_TRITON` | `1` | 启用/禁用 Triton (`0` 禁用) |
| `MINNLP_TRITON_BACKEND` | `auto` | 强制指定后端 (`ascend`, `nvidia`, `cpu`) |

## 模块结构

```
mindnlp.triton/
├── __init__.py                 # 主入口导出
├── README.md                   # 本文档
├── kernels/
│   ├── __init__.py
│   ├── activations.py          # Triton GELU/SwiGLU 实现
│   ├── benchmark.py            # 性能测试工具
│   └── mindspore_adapter.py    # MindSpore 适配层 (MSGELU, MSSwiGLU)
├── backends/
│   ├── __init__.py
│   ├── detect.py               # 后端自动检测
│   └── ascend.py               # Ascend NPU 支持
├── integration/
│   ├── __init__.py
│   └── mindtorch_v2.py         # mindtorch_v2 集成
├── pipeline/
│   ├── __init__.py            # run_pipeline, run_all
│   ├── __main__.py            # CLI 入口
│   ├── profiling.py           # Phase 1: 性能分析
│   ├── testing.py             # Phase 2: 精度验证
│   ├── benchmark.py            # Phase 3: 算子对比
│   ├── e2e.py                 # Phase 4: 端到端测试
│   ├── report.py              # Phase 5: 报告生成
│   └── runner.py              # 管线调度器
└── docs/
    ├── ANALYSIS_REPORT.md      # 瓶颈分析报告
    └── PERFORMANCE_REPORT.md  # 性能测试报告
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

## Profiling 数据

Qwen2.5-0.5B 模型各算子时间分布：

| 算子 | 时间 (ms) | 占比 | 类型 |
|------|-----------|------|------|
| down_proj | 4383.5 | 30.5% | GEAM |
| gate_proj | 3862.2 | 26.8% | GEAM |
| up_proj | 3831.0 | 26.6% | GEAM |
| act_fn | 421.2 | 2.9% | Elementwise |
| RMSNorm | 143.1 | 1.0% | RMSNorm |

**关键发现**: MLP 层占 86.9%，但其中激活函数仅占 2.9%

## 注意事项

1. **Triton-Ascend 限制**：
   - Grid 限制: coreDim <= 65535
   - 不支持 `continue` 语句
   - 大型 kernel 可能编译不稳定

2. **自动降级**：当 Triton 不可用时，自动回退到原生实现

3. **兼容性问题**：在 MindSpore 2.8.0 + torch 2.7.1+npu 环境下，Pipeline 部分阶段可能出现兼容性问题。核心 kernels 在 NPU 上验证正常工作。

## 许可证

Apache 2.0 - 与 MindNLP 保持一致