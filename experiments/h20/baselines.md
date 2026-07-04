# H20 MVP Baseline 设计

## A0: 无知识库 Coding Agent

输入：OpSpec、reference、harness 命令、预算限制。

不提供任何历史知识、优化规则、文档摘要或示例。用于衡量 Agent 原生能力。

## A1: 普通 RAG / 向量库增强 Agent

输入：OpSpec、reference、harness 命令、预算限制、普通文档检索片段。

RAG 数据包括 Triton/PyTorch/CUDA 公开文档片段和调研报告片段。检索仅按文本相似度，不做 phase-aware、evidence-aware、gate-aware 过滤。用于验证“普通 RAG 是否足够”。

## A2: 简单规则库/经验库增强 Agent

输入：OpSpec、reference、harness 命令、预算限制、静态规则列表。

规则包括 coalescing、tiling、shared memory、occupancy、mask、fp32 accumulation 等常见建议。规则没有证据、版本、失败边界、promotion gate，也不随实验演化。用于验证“经验规则堆叠是否足够”。

## A3: ECC-KB 增强 Agent

输入：OpSpec、reference、harness 命令、预算限制、ECC-KB ContextPacket。

ContextPacket 分阶段返回：

- `must_obey`：语义和合法性硬约束；
- `recommended_actions`：可执行动作；
- `anti_actions`：禁做或高风险动作；
- `validation_plan`：当前阶段必须通过的门控；
- `evidence_refs`：用于审计和压缩解释；
- `stop_conditions`：停止无效搜索的条件。

A3 会写回 evolution records，但 Round-1 中 A3 的同轮任务不得使用未来任务知识。Round-2 使用经过 promotion gate 的 v1 KB。

## 公平性要求

所有组必须固定：

```text
same agent
same model if applicable
same temperature/search policy if configurable
same max iterations
same token budget
same wall-clock budget
same GPU benchmark budget
same task order randomization
same harness and hidden tests
```

若某 Agent 不能报告 token，至少记录 prompt/output character count 和 wall-clock time。
