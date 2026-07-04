# H20 MVP 算子任务集

## 1. 任务选择原则

MVP 任务需要中等复杂度、优化空间明确、reference 容易构造、错误模式丰富，并且能在 H20 上以 Triton/PyTorch 运行。第一批不选超复杂融合 attention，也不选过于 trivial 的 elementwise。

## 2. 算子清单

### 2.1 Row reduction

目标：实现 `y = reduce(x, axis=-1)`，包含 `sum` 和 `max` 两种变体。

Shapes：

```text
[1024, 64]
[1024, 257]
[4096, 1024]
[8192, 4096]
[128, 8192]
```

DTypes：`fp32`, `fp16`, `bf16`。`sum` 的累加精度需按 OpSpec 指定，默认 fp32 accumulation。

覆盖问题：tail mask、非 2 次幂、reduction order、accumulation dtype、memory bandwidth、block size tuning。

### 2.2 Row softmax

目标：实现 `torch.softmax(x, dim=-1)`。

Shapes：

```text
[1024, 128]
[1024, 257]
[4096, 1024]
[8192, 2048]
[512, 4096]
```

DTypes：`fp16`, `bf16`, `fp32`。默认输出 dtype 与输入一致，内部可 fp32 计算。

覆盖问题：max subtraction、exp/sum/divide 语义、数值稳定、mask、profile after correctness、row tiling。

### 2.3 LayerNorm forward

目标：实现 `y = (x - mean) / sqrt(var + eps) * gamma + beta`，axis=-1。

Shapes：

```text
[1024, 768]
[2048, 1024]
[4096, 2048]
[8192, 4096]
[1024, 8192]
```

DTypes：`fp16`, `bf16` input；accumulation fp32；`eps = 1e-5`。

覆盖问题：two reductions、epsilon、broadcast gamma/beta、variance formula、register pressure、hidden size tuning。

### 2.4 Matmul variants

目标：实现 `C = A @ B`，可选加 bias 或 activation 不放入第一轮。

Shapes：

```text
M,N,K = 128, 128, 128
M,N,K = 512, 512, 512
M,N,K = 1024, 1024, 1024
M,N,K = 4096, 256, 1024
M,N,K = 256, 4096, 1024
```

DTypes：`fp16`, `bf16` input；accumulation fp32 or backend default as declared。

覆盖问题：tile size、num_warps、stages、memory coalescing、compute vs memory bound、baseline 强度。

## 3. Round 划分

Round-1 用公开 shape 构建初始知识与轨迹。Round-2 使用 held-out variants：

```text
softmax: [2048, 513], [16384, 1024]
layernorm: [4096, 3072], [2048, 6144]
reduction: [2048, 1536], [16384, 255]
matmul: [768, 768, 768], [2048, 512, 1536]
```

Round-2 的 shape 不用于知识构建，只用于验证自我进化。
