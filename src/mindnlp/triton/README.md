# MindNLP Triton 模块

基于 MindSpore 框架的高性能 Triton 算子实现，用于 Ascend NPU 和 NVIDIA GPU。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [性能数据](#性能数据)
- [模块结构](#模块结构)
- [API 详细使用](#api-详细使用)
- [优化管线](#优化管线-pipeline)
- [基准测试](#基准测试)
- [适用模型](#适用模型)
- [注意事项](#注意事项)

---

## 环境要求

### 硬件环境

| 环境 | 要求 |
|------|------|
| 服务器 | Atlas 800I A2 推理服务器 或 Atlas 800T A2 训练服务器 |
| NPU | Ascend 910B4 或类似型号 |

### 软件环境

| 软件 | 版本要求 |
|------|----------|
| 操作系统 | openEuler 或 Ubuntu Linux |
| Python | ≥ 3.9 且 < 3.12 |
| CANN | ≥ 8.1.RC1（推荐 8.3.RC1） |
| MindSpore | ≥ 2.7.0 |
| Triton | ≥ 3.2.0（Ascend 适配版） |

### 安装依赖

```bash
# 安装 Triton (Ascend 版本)
pip install triton

# 或者从源码安装
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e .
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINNLP_TRITON` | `1` | 启用/禁用 Triton (`0` 禁用) |
| `MINNLP_TRITON_BACKEND` | `auto` | 强制指定后端 (`ascend`, `nvidia`, `cpu`) |
| `TRITON_BACKEND` | `mindspore` | Triton 后端策略（建议设为 `mindspore`） |

---

## 快速开始

### 1. 基本导入

```python
# 设置环境变量（建议在导入前设置）
import os
os.environ['TRITON_BACKEND'] = 'mindspore'

# MindSpore 方式（推荐）
import mindspore as ms
from mindnlp.triton import MSGELU, MSSwiGLU, gelu, swiglu

# PyTorch 兼容方式
import torch
from mindnlp.triton import TritonGELU, TritonSwiGLU, triton_gelu, triton_swiglu
```

### 2. 基础使用

```python
import torch
import os
os.environ['TRITON_BACKEND'] = 'mindspore'

from mindnlp.triton import triton_gelu, triton_swiglu

# 创建输入张量 (NPU)
x = torch.randn(24, 512, 4864, device='npu')
gate = torch.randn(24, 512, 4864, device='npu')
up = torch.randn(24, 512, 4864, device='npu')

# 使用 Triton GELU
output_gelu = triton_gelu(x)
print(f"GELU output shape: {output_gelu.shape}")

# 使用 Triton SwiGLU
output_swiglu = triton_swiglu(gate, up)
print(f"SwiGLU output shape: {output_swiglu.shape}")
```

---

## 性能数据

**实测结果 (Ascend NPU, 公平对比)**：

| 算子 | 数据规模 | PyTorch Native | Triton | 加速比 | 说明 |
|------|----------|----------------|--------|--------|------|
| **SwiGLU** | 24×512×4864 | 3.06ms | 1.19ms | **2.58x** | ✅ 推荐使用 |
| GELU | 24×512×4864 | 0.75ms | 0.94ms | 0.80x | ❌ 不推荐 |

**测试条件**：
- 公平对比：Triton 和 PyTorch Native 都在 NPU 上运行
- 基准函数：`torch.nn.functional.gelu` / `torch.nn.functional.silu`
- 数值精度：GELU ≈ Native, SwiGLU ≈ Native

---

## 模块结构

```
mindnlp.triton/
├── __init__.py                 # 主入口，导出所有公开 API
├── README.md                   # 本文档
├── kernels/
│   ├── __init__.py
│   ├── activations.py          # Triton GELU/SwiGLU 实现
│   ├── benchmark.py           # 性能测试工具
│   └── mindspore_adapter.py   # MindSpore 适配层 (MSGELU, MSSwiGLU)
├── backends/
│   ├── __init__.py
│   ├── detect.py              # 后端自动检测
│   └── ascend.py             # Ascend NPU 支持
├── integration/
│   ├── __init__.py
│   └── mindtorch_v2.py       # mindtorch_v2 集成
├── pipeline/
│   ├── __init__.py           # run_pipeline, run_all
│   ├── __main__.py           # CLI 入口
│   ├── runner.py             # 管线调度器
│   ├── profiling.py          # Phase 1: 性能分析
│   ├── testing.py            # Phase 2: 精度验证
│   ├── benchmark.py           # Phase 3: 算子对比
│   ├── e2e.py               # Phase 4: 端到端测试
│   └── report.py             # Phase 5: 报告生成
└── docs/
    ├── ISSUE_DRAFT.md         # Issue 提交草稿
    ├── ANALYSIS_REPORT.md      # 瓶颈分析报告
    └── PERFORMANCE_REPORT.md  # 性能测试报告
```

---

## API 详细使用

### 1. Triton 激活函数 (PyTorch 兼容)

```python
from mindnlp.triton import triton_gelu, triton_swiglu

# GELU 激活
x = torch.randn(batch, seq_len, hidden_dim, device='npu')
output = triton_gelu(x)

# SwiGLU 激活 (gate, up)
gate = torch.randn(batch, seq_len, intermediate_dim, device='npu')
up = torch.randn(batch, seq_len, intermediate_dim, device='npu')
output = triton_swiglu(gate, up)
```

### 2. nn.Module 接口

```python
from mindnlp.triton import TritonGELU, TritonSwiGLU

# 作为 nn.Module 使用
model = TritonGELU()
output = model(x)

model = TritonSwiGLU()
output = model(gate, up)
```

### 3. MindSpore Cell 接口

```python
from mindnlp.triton import MSGELU, MSSwiGLU

# MindSpore Cell 方式
gelu_cell = MSGELU()
output = gelu_cell(x)

swiglu_cell = MSSwiGLU()
output = swiglu_cell(gate, up)
```

### 4. 自动后端选择

```python
from mindnlp.triton import gelu, swiglu

# 根据环境自动选择 Triton 或 Native 实现
output = gelu(x)      # 自动选择
output = swiglu(gate, up)  # 自动选择
```

### 5. 基准测试 API

```python
from mindnlp.triton.kernels.benchmark import (
    benchmark_function,
    benchmark_activation,
    benchmark_swiglu,
    compare_activations,
)

# 单函数基准测试
result = benchmark_function(
    func=triton_gelu,
    x,
    warmup=10,
    runs=100
)
print(f"Mean time: {result['mean']*1000:.3f}ms")

# 激活函数基准测试
result = benchmark_activation(
    activation_fn=triton_gelu,
    shape=(24, 512, 4864),
    device='npu'
)

# SwiGLU 基准测试
result = benchmark_swiglu(
    shape=(24, 512, 4864),
    device='npu'
)
print(f"Speedup: {result['speedup']:.2f}x")

# 对比所有激活函数
results = compare_activations(
    shape=(24, 512, 4864),
    device='npu'
)
print(f"GELU speedup: {results['gelu']['speedup']:.2f}x")
print(f"SwiGLU speedup: {results['swiglu']['speedup']:.2f}x")
```

### 6. 公平基准测试 (推荐)

```python
from mindnlp.triton.kernels.benchmark import run_fair_benchmark

# 运行完整公平基准测试
results = run_fair_benchmark(
    shape=(24, 512, 4864),
    device='npu',
    warmup=10,
    runs=100
)

print(f"GELU speedup: {results['summary']['gelu_speedup']}")
print(f"SwiGLU speedup: {results['summary']['swiglu_speedup']}")
```

---

## 优化管线 (Pipeline)

完整的 Qwen 模型 Triton 优化管线，包含 5 个阶段：

### 1. Python API

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

# 运行指定阶段
results = run_pipeline(config, ['profiling', 'benchmark', 'e2e', 'report'])

# 运行全部阶段
results = run_all(config)
```

### 2. CLI 使用

```bash
# 运行全部阶段
python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --phase all

# 运行指定阶段
python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --phase profiling,benchmark

# 指定设备
python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --device npu --phase all

# 输出到文件
python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --phase all --output results.json
```

### 3. 管线阶段说明

| 阶段 | 说明 | 输出 |
|------|------|------|
| `profiling` | 分析 Qwen 模型性能数据，识别瓶颈算子 | 瓶颈分析报告 |
| `test` | 验证 Triton kernel 数值精度 (阈值 < 1e-5) | 精度验证结果 |
| `benchmark` | 单算子性能对比 (Triton vs Native) | 加速比数据 |
| `e2e` | 端到端 MLP 性能验证 | 端到端性能 |
| `report` | 生成汇总报告和优化建议 | 完整报告 |

---

## 基准测试

### 1. 基本基准测试

```python
import torch
import time
from mindnlp.triton import triton_gelu, triton_swiglu

def benchmark(func, x, iterations=100, warmup=10):
    """基准测试辅助函数"""
    for _ in range(warmup):
        func(x)
        if x.is_npu:
            torch.npu.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(x)
        if result.is_npu:
            torch.npu.synchronize()
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms

# 测试 GELU
x = torch.randn(24, 512, 4864, device='npu')
gelu_time = benchmark(triton_gelu, x)
print(f"Triton GELU: {gelu_time:.3f}ms")
```

### 2. 公平对比测试

```python
import torch
import time

def fair_benchmark(shape, device='npu', iterations=100):
    """公平对比测试：Triton vs PyTorch Native"""
    from mindnlp.triton import triton_gelu, triton_swiglu

    x = torch.randn(*shape, device=device)
    gate = torch.randn(*shape, device=device)
    up = torch.randn(*shape, device=device)

    # Warmup
    for _ in range(10):
        triton_gelu(x); torch.nn.functional.gelu(x)
        triton_swiglu(gate, up); gate * torch.nn.functional.silu(gate) * up
    torch.npu.synchronize()

    # GELU benchmark
    t0 = time.perf_counter()
    for _ in range(iterations):
        torch.nn.functional.gelu(x)
    torch.npu.synchronize()
    native_g_time = (time.perf_counter() - t0) * 1000 / iterations

    t0 = time.perf_counter()
    for _ in range(iterations):
        triton_gelu(x)
    torch.npu.synchronize()
    triton_g_time = (time.perf_counter() - t0) * 1000 / iterations

    # SwiGLU benchmark
    t0 = time.perf_counter()
    for _ in range(iterations):
        gate * torch.nn.functional.silu(gate) * up
    torch.npu.synchronize()
    native_s_time = (time.perf_counter() - t0) * 1000 / iterations

    t0 = time.perf_counter()
    for _ in range(iterations):
        triton_swiglu(gate, up)
    torch.npu.synchronize()
    triton_s_time = (time.perf_counter() - t0) * 1000 / iterations

    return {
        'gelu_speedup': native_g_time / triton_g_time,
        'swiglu_speedup': native_s_time / triton_s_time,
    }

# 运行公平对比
results = fair_benchmark((24, 512, 4864), 'npu')
print(f"GELU speedup: {results['gelu_speedup']:.2f}x")
print(f"SwiGLU speedup: {results['swiglu_speedup']:.2f}x")
```

### 3. 数值精度验证

```python
import torch
from mindnlp.triton import triton_gelu, triton_swiglu

def verify_accuracy(shape=(24, 512, 4864), device='npu'):
    """验证 Triton 实现与 PyTorch Native 的精度一致性"""
    x = torch.randn(*shape, device=device)
    gate = torch.randn(*shape, device=device)
    up = torch.randn(*shape, device=device)

    # GELU 精度
    triton_g = triton_gelu(x).cpu().float()
    native_g = torch.nn.functional.gelu(x).cpu().float()
    gelu_diff = float(torch.max(torch.abs(triton_g - native_g)))

    # SwiGLU 精度
    triton_s = triton_swiglu(gate, up).cpu().float()
    native_s = (gate * torch.nn.functional.silu(gate) * up).cpu().float()
    swiglu_diff = float(torch.max(torch.abs(triton_s - native_s)))

    print(f"GELU  max diff: {gelu_diff:.2e}")
    print(f"SwiGLU max diff: {swiglu_diff:.2e}")
    print(f"GELU  PASS: {gelu_diff < 1e-4}")
    print(f"SwiGLU PASS: {swiglu_diff < 1e-4}")

verify_accuracy()
```

---

## 适用模型

### 推荐使用 Triton 的模型

| 模型 | 激活函数 | 推荐优化 | 加速比 |
|------|----------|----------|--------|
| Qwen2 (0.5B) | SwiGLU | `MSSwiGLU` / `triton_swiglu` | **2.58x** |
| Qwen2.5 | SwiGLU | `MSSwiGLU` / `triton_swiglu` | **2.58x** |
| BLOOM | GELU | ❌ 不推荐 | 0.80x |
| LLaMA | GELU | ❌ 不推荐 | 0.80x |

### 不建议优化的算子

| 算子 | 原因 | 建议 |
|------|------|------|
| GEAM (matmul) | CANN 快 17x | 使用 CANN 原生 |
| RMSNorm | CANN 快 100x | 使用 CANN 原生 |
| LayerNorm | CANN 快 100x | 使用 CANN 原生 |
| Attention/Softmax | CANN 快 5x | 使用 CANN 原生 |

---

## 注意事项

### 1. Triton-Ascend 限制

- **Grid 限制**: `coreDim <= 65535`
- **不支持 `continue` 语句**: 需要用 while 循环替代
- **不支持 `tl.extract_slice/insert_slice`**: 需要简化算法
- **编译不稳定**: 大型 kernel 可能导致 SIGSEGV

### 2. 环境变量配置

```bash
# 推荐的环境变量配置
export TRITON_BACKEND=mindspore
export MINNLP_TRITON=1
export MINNLP_TRITON_BACKEND=auto
```

### 3. 自动降级

当 Triton 不可用时，模块会自动回退到原生 PyTorch 实现：

```python
from mindnlp.triton import gelu, swiglu

# 自动选择可用实现
output = gelu(x)  # 如果 Triton 不可用，自动使用 PyTorch Native
```

### 4. 常见问题

**Q: `torch.randn` 挂起怎么办？**
```python
# 设置环境变量
os.environ['TRITON_BACKEND'] = 'mindspore'
# 在导入 mindnlp.triton 之前设置
```

**Q: 性能测试结果不稳定？**
- 确保 NPU 处于稳定状态
- 增加 warmup 和 iterations
- 多次测试取平均值

**Q: 精度验证失败？**
- 检查输入数据类型是否为 float32
- 确保对比的两边都在同一设备上

---

## 许可证

Apache 2.0 - 与 MindNLP 保持一致

---

## 相关文档

- [瓶颈分析报告](./docs/ANALYSIS_REPORT.md)
- [性能测试报告](./docs/PERFORMANCE_REPORT.md)
- [Issue 提交草稿](./docs/ISSUE_DRAFT.md)
