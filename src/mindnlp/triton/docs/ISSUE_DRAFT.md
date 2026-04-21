# Qwen 模型 Triton 算子优化 - Issue 内容草稿

## Issue 标题
[MindNLP] Qwen 模型 Triton 激活函数优化 (SwiGLU 2.58x 加速)

## Issue 内容

### 背景
基于 MindSpore 框架，使用 Triton 技术栈为 Qwen 模型开发高性能替换算子。

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

### 测试的算子列表

| 算子 | 类型 | 加速比 | 结论 |
|------|------|--------|------|
| **SwiGLU** | 激活函数 | **2.58x** | ✅ 推荐使用 Triton |
| GELU | 激活函数 | 0.80x | ❌ PyTorch 更快 |
| GEAM (matmul) | 矩阵乘法 | 0.06x | ❌ CANN 更快 (17x) |
| RMSNorm | 归一化 | 0.01x | ❌ CANN 更快 |
| Flash Attention | 注意力 | 0.18x | ❌ CANN 更快 |

### 性能数据 (Ascend NPU, 公平对比)

**测试配置**: (24, 512, 4864) - Qwen2-0.5B 24层批量

| 算子 | PyTorch Native | Triton | 加速比 |
|------|----------------|--------|--------|
| **SwiGLU** | 3.06ms | 1.19ms | **2.58x** |
| GELU | 0.75ms | 0.94ms | 0.80x |

**测试方法**：Triton 和 PyTorch Native 都在 NPU 上运行，确保公平对比

### 优化建议

1. **对于使用 SwiGLU 的模型**（如 Qwen2、Qwen2.5）：
   - 使用 `triton_swiglu()` 或 `swiglu()`
   - 预期激活函数加速 **2.58x**

2. **不要尝试用 Triton 替代 CANN matmul**：
   - 性能差距太大 (17x)

### 集成状态

代码已集成到 `mindnlp/src/mindnlp/triton/`：
- `kernels/activations.py` - Triton GELU/SwiGLU 实现
- `kernels/mindspore_adapter.py` - MindSpore 适配层
- `pipeline/` - 优化管线

### 适用模型

| 模型 | 激活函数 | 推荐优化 | 加速比 |
|------|----------|----------|--------|
| Qwen2 (0.5B) | SwiGLU | ✅ triton_swiglu | **2.58x** |
| Qwen2.5 | SwiGLU | ✅ triton_swiglu | **2.58x** |
| LLaMA | GELU | ❌ 不推荐 | 0.80x |