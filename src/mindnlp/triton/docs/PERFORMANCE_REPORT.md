# Triton Qwen2-0.5B 算子优化性能报告

## 1. 任务概述

**任务目标**：基于 MindSpore 框架，使用 Triton 技术栈为 Qwen2-0.5B 模型开发高性能替换算子。

**验收标准**：
1. 算子分析报告：提交 Issue 到 mindnlp 仓库
2. 源码提交：完整的 Triton 算子实现代码
3. 性能数据：详细的性能测试对比报告，标明加速比

## 2. Profiling 分析结果

### 2.1 Qwen2-0.5B 瓶颈分析

根据 MindSpore Profiler 数据：

| 算子 | 总时间 (ms) | 占比 | 类型 |
|------|-------------|------|------|
| down_proj | 4383.5 | 30.5% | GEAM |
| gate_proj | 3862.2 | 26.8% | GEAM |
| up_proj | 3831.0 | 26.6% | GEAM |
| q_proj | 754.8 | 5.2% | GEAM |
| o_proj | 727.4 | 5.1% | GEAM |
| act_fn (silu) | 421.2 | 2.9% | Elementwise |
| k_proj | 141.5 | 1.0% | GEAM |
| v_proj | 123.3 | 0.9% | GEAM |
| post_attention_layernorm | 75.5 | 0.5% | RMSNorm |
| input_layernorm | 67.6 | 0.5% | RMSNorm |

**关键发现**：
- GEAM 算子占 **96%** 总时间
- MLP 层 (gate_proj + up_proj + down_proj + act_fn) 占 **86.9%** 时间
- 其中 act_fn (激活函数) 只占 **2.9%**
- CANN matmul 比 Triton matmul 快 **17x**

### 2.2 优化策略分析

| 算子类型 | 占时间比 | CANN优化程度 | Triton适用性 |
|----------|----------|--------------|-------------|
| GEAM (matmul) | 96% | 极度优化 (17x) | ❌ 不适用 |
| Elementwise (激活) | 2.9% | 一般 | ✅ 适用 |

## 3. Triton 算子测试结果

### 3.1 测试环境

- PyTorch: 2.7.1+npu
- Triton-Ascend: 3.2.0
- MindSpore: 2.8.0
- CANN: 8.5.0
- Device: Ascend NPU

### 3.2 实测结果 (Ascend NPU, 公平对比)

**配置**: (24, 512, 4864) - Qwen2-0.5B 24层批量
**测试条件**: Triton 和 PyTorch Native 都在 NPU 上运行

| 算子 | PyTorch Native | Triton | 加速比 | 结论 |
|------|----------------|--------|--------|------|
| **GELU** | 0.75ms | 0.94ms | **0.80x** | PyTorch 更快 |
| **SwiGLU** | 3.06ms | 1.19ms | **2.58x** | Triton 更快 |

**数值精度**：
- GELU: max diff ≈ 0 (与 native exact 匹配)
- SwiGLU: max diff ≈ 0 (与 native 匹配)

**注**：
- 早期报告中的 41.68x/7.16x 加速比是基于 NPU Triton vs CPU Native 的不公平对比
- 公平对比结果：GELU 0.80x, SwiGLU 2.58x

### 3.3 失败方案 (加速比 < 1，已剔除)

| 算子 | 测试结果 | 原因 |
|------|----------|------|
| Flash Attention | 0.18x | CANN attention 已极度优化 |
| MLP Matmul | 0.02x | CANN matmul 比 Triton 快 17x |
| SILU | 0.95x | CANN 已优化 |
| Add/Mul | ~1x | CANN 已优化 |
| RMSNorm | N/A | 测试代码有 bug |

### 3.4 Triton-Ascend 限制

1. **不支持 `continue` 语句**：需要用 while 循环替代
2. **不支持 `tl.extract_slice/insert_slice`**：需要简化算法
3. **Grid 限制**：coreDim <= 65535
4. **编译不稳定**：大型 kernel 可能导致 SIGSEGV

## 4. 适用场景

| 模型 | 激活函数 | Triton 优化 | 加速比 | 推荐 |
|------|----------|-------------|--------|------|
| Qwen2 (0.5B) | SwiGLU | ✅ triton_swiglu | **2.58x** | ✅ 推荐 |
| Qwen2.5 | SwiGLU | ✅ triton_swiglu | **2.58x** | ✅ 推荐 |
| LLaMA | GELU | ⚠️ triton_gelu | **0.80x** | ❌ 不推荐 |
| 标准 MLP | silu | ❌ 无收益 | CANN 已优化 | ❌ 不推荐 |

## 5. 结论与建议

### 5.1 结论

1. **数值精度正确**：Triton GELU 和 SwiGLU 与 PyTorch Native 实现匹配
2. **SwiGLU 推荐使用 Triton**：公平对比加速比 **2.58x**
3. **GELU 不推荐使用 Triton**：公平对比加速比 **0.80x** (PyTorch 更快)
4. **CANN matmul 不可超越**：Triton matmul 比 CANN 慢 17x
5. **基准测试方法重要**：早期 41.68x/7.16x 是 NPU vs CPU 的不公平对比

### 5.2 建议

1. **对于使用 SwiGLU 的模型**（如 Qwen2、Qwen2.5）：
   - 使用 `triton_swiglu()` 或 `swiglu()`
   - 预期激活函数加速 **2.58x**

2. **对于使用 GELU 的模型**（如 LLaMA）：
   - 使用 PyTorch Native `torch.nn.functional.gelu` 性能更好

3. **不要尝试用 Triton 替代 CANN matmul**：
   - 性能差距太大 (17x)
   - CANN 已极度优化

## 6. 集成状态

已集成到 mindnlp 仓库：
- 路径：`src/mindnlp/triton/`
- 包含：kernels, backends, pipeline, integration
- 支持：Ascend NPU, NVIDIA GPU

## 7. 附录：使用示例

```bash
# 导入 Triton 激活函数
from mindnlp.triton import MSGELU, MSSwiGLU, gelu, swiglu

# MindSpore Cell 方式
act = MSGELU()
output = act(x)

# 函数方式
output = gelu(x)

# SwiGLU (gate, up)
output = swiglu(gate, up)

# 环境变量禁用 Triton
export MINNLP_TRITON=0
```