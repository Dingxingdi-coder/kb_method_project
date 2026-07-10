# H20 MVP Baseline 设计

本文定义 H20 MVP 首轮只执行的四个主实验组。实验目标是拆分三类增益来源：原始资料增益、知识化表示增益、ECC-KB 方法增益。

## 0. 共同前提

所有带检索的实验组必须从同一批冻结资料派生：

```text
raw_corpus_v0
  -> KB v0
```

`raw_corpus_v0` 来自公开文档、论文、调研报告、示例代码、已有 kernel、benchmark 说明和工具链资料。`KB v0` 只能从 `raw_corpus_v0` 抽取，不能额外加入 A1 不可见的资料。

实验运行期间不使用实时联网搜索。每次 run 必须记录：

```text
source_corpus_version
raw_corpus_index_version
kb_version
kb_plain_rag_index_version
ecc_kb_index_version
retrieved_context_length
retrieved_item_ids
```

## A0: `A0_prompt`

输入：OpSpec、reference、harness 命令。预算使用由 Hook / trace 事后统计，并按统一审计阈值标记。

A0 不提供任何历史知识、资料库片段、知识库片段、优化规则、文档摘要或示例。A0 衡量 Coding Agent 在固定 harness 下的原生能力。

## A1: `A1_raw_corpus_rag`

输入：OpSpec、reference、harness 命令、从 `raw_corpus_v0` 普通 RAG 召回的原始资料 chunk。预算使用由 Hook / trace 事后统计，并按统一审计阈值标记。

A1 的召回对象是原始资料，而不是 `kb/`。典型 chunk 包括：

- Triton、PyTorch、CUDA 或 profiling 文档段落；
- 自动算子生成论文或调研报告段落；
- 示例代码说明；
- benchmark、harness 或 correctness 说明。

A1 只使用普通文本相似度召回。A1 不使用 phase-aware 召回、evidence-aware 重排、gate-aware 过滤、结构化 capsule 解析、演化写回或 ECC-KB ContextPacket。

A1 用于验证“把原始资料直接 RAG 给 Agent 是否足够”。

## A2: `A2_kb_plain_rag`

输入：OpSpec、reference、harness 命令、从 `KB v0` 普通 RAG 召回的知识单元文本。预算使用由 Hook / trace 事后统计，并按统一审计阈值标记。

A2 的召回对象是已经从 `raw_corpus_v0` 转换出的知识库，而不是原始资料。A2 可以看到知识单元的文本化内容，例如 capsule 的适用条件、建议动作、预期效果、证据摘要和失败边界。

A2 必须保持“普通 RAG”性质。A2 不使用阶段化召回、硬过滤、合法性过滤、证据强度重排、ContextPacket、stop capsule 或状态写回。

A2 用于验证“资料被转换成结构化知识后，即使只做普通 RAG，是否已经比原始资料 RAG 更有效”。

## A3: `A3_ecc_kb`

输入：OpSpec、reference、harness 命令、ECC-KB ContextPacket。预算使用由 Hook / trace 事后统计，并按统一审计阈值标记。

A3 使用完整 ECC-KB 方法。ContextPacket 分阶段返回：

- `must_obey`：语义和合法性硬约束；
- `recommended_actions`：可执行动作；
- `anti_actions`：禁做或高风险动作；
- `validation_plan`：当前阶段必须通过的门控；
- `evidence_refs`：用于审计和压缩解释；
- `stop_conditions`：停止无效搜索的条件。

A3 会写回 evolution records。Round-1 中，A3 的同轮任务不得使用未来任务知识。Round-1 结束后，A3 轨迹先进入 quarantine，再由 promotion gate 决定是否进入后续 `KB v1`。

A3 用于验证“在同一份资料和同一个知识库基础上，ECC-KB 的阶段化召回、证据门控、ContextPacket、stop condition 和演化机制是否带来额外收益”。

## 四组差异汇总

```text
A0_prompt:
  source: none
  kb: none
  retrieval: none
  ecc_method: none
  writeback: none

A1_raw_corpus_rag:
  source: raw_corpus_v0
  kb: none
  retrieval: plain_text_rag
  ecc_method: none
  writeback: none

A2_kb_plain_rag:
  source: derived_from_raw_corpus_v0
  kb: KB v0 as text
  retrieval: plain_text_rag
  ecc_method: none
  writeback: none

A3_ecc_kb:
  source: derived_from_raw_corpus_v0
  kb: KB v0 as executable evidence capsules
  retrieval: phase_and_evidence_aware_context_packet
  ecc_method: enabled
  writeback: quarantine_then_promotion
```

## 公平性要求

所有组必须固定：

```text
same agent
same model if applicable
same temperature/search policy if configurable
same post-hoc audit threshold for candidate count / harness calls
same post-hoc audit threshold for wall-clock time
same task order randomization
same harness and hidden tests
same raw_corpus_v0 for A1/A2/A3 derivation
same KB v0 for A2/A3
```

A1、A2、A3 的 retrieved context length 上限必须相同。若某组召回结果超过上限，检索器必须截断，而不是放宽上限。token 相关信息不进入 pilot 自动指标；若需要，只从 Agent/Hook 日志人工读取。
