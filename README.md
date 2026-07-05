# ECC-KB: 面向自动算子生成的可执行证据胶囊知识库

本仓库是一个 repo-ready 的方法包，用于构建和演化“后端芯片无关、Coding Agent 无关”的自动算子生成知识库。它不是普通 RAG、普通知识图谱或经验规则库，而是一个以 **可验证知识单元** 为核心的执行型知识系统。

方法中文名：**可执行证据胶囊知识库**。英文名：**Executable Evidence-Capsule Knowledge Base, ECC-KB**。

## 核心问题

自动算子生成的失败通常不是单一代码错误，而是搜索空间、语义边界、硬件约束、验证协议、profile 归因和历史经验共同失配。普通文档检索只能提高“能写出类似代码”的概率，不能稳定提高：

- 编译成功率；
- 数值正确率；
- 相对强 baseline 的性能；
- 达到目标所需迭代次数；
- time/token 成本；
- 对新 shape、dtype、后端的迁移能力。

ECC-KB 把知识定义为可执行、可验证、可失效、可迁移的条件化动作单元，而不是文档 chunk。每个知识单元都必须携带适用条件、作用机制、证据、执行建议、验证门控、失败边界和后端抽象映射。

## 方法贡献

1. **知识对象重定义**：知识不是文本片段，而是 `condition -> action -> expected_effect -> evidence -> gate -> boundary` 的可执行证据胶囊。
2. **双层后端解耦**：将后端无关的抽象机制层与 Nvidia H20 / AMD / 华为等后端实例层分离。
3. **阶段化知识服务**：生成、编译修复、正确性修复、性能优化、autotune、复盘写回使用不同知识包，避免把噪声塞进上下文。
4. **自我进化闭环**：通过 quarantine、promotion、stale、reject 状态机沉淀成功和失败轨迹，并阻止错误经验污染稳定库。
5. **H20 可验证 MVP**：第一阶段只需要 Nvidia H20 环境，通过 Triton + PyTorch reference + 固定 harness 验证方法是否降低 time/token/iteration 并提高正确率和性能。

## H20 MVP 对照组

H20 首轮实验先只跑四组。所有带检索的组都从同一个冻结资料库 `raw_corpus_v0` 派生，避免因为资料量不同造成不公平比较。

| 组别 | 输入 | 验证的问题 |
|---|---|---|
| `A0_prompt` | 普通 prompt + OpSpec + harness 指令 | Coding Agent 原生能力 |
| `A1_raw_corpus_rag` | 对 `raw_corpus_v0` 做普通 RAG，返回原始资料 chunk | 原始资料直接 RAG 是否足够 |
| `A2_kb_plain_rag` | 对由 `raw_corpus_v0` 转换出的 `KB v0` 做普通 RAG，返回知识单元文本 | 只做知识化表示是否已经有效 |
| `A3_ecc_kb` | 使用 ECC-KB ContextPacket、阶段化召回、证据重排和演化写回 | ECC-KB 方法层是否带来额外收益 |

`A2_kb_plain_rag` 不使用阶段化召回、证据门控、ContextPacket、stop capsule 或写回机制。`A3_ecc_kb` 才使用这些方法能力。

## 仓库结构

```text
kb_method_project/
  README.md
  docs/
    method.md
    h20_mvp_protocol.md
    migration.md
    validation_guide.md
  schemas/
    knowledge_unit.schema.json
    evolution_record.schema.json
  sources/
    README.md
    raw_corpus/
    registry/
    derived_claims/
  experiments/
    h20/
      operators.md
      baselines.md
      metrics.md
      evolution_loop.md
  examples/
    operator_walkthrough.md
  protocols/
    agent_interface.md
  kb/
    stable/
    quarantine/
    failures/
    evidence/
  tools/
    README.md
```

## 推荐阅读顺序

先读 `docs/method.md` 判断方法是否有新意。再读 `docs/h20_mvp_protocol.md` 判断能否在 H20 上验证。然后读 `experiments/h20/baselines.md` 判断四组对照实验是否公平。再读 `schemas/knowledge_unit.schema.json` 和 `schemas/evolution_record.schema.json` 判断知识是否可机器消费。最后读 `experiments/h20/*` 和 `examples/operator_walkthrough.md` 判断实验是否可落地。

## 最小落地路径

第一周冻结 `raw_corpus_v0`，建立 source registry，并从同一资料库转换出 `KB v0`。第二周实现 schema 校验、run 记录、H20 harness、普通 RAG 检索器和 ECC-KB ContextPacket 生成器。第三周在相同 agent、相同任务、相同 token/wall-clock/GPU benchmark 预算下跑 `A0_prompt`、`A1_raw_corpus_rag`、`A2_kb_plain_rag` 和 `A3_ecc_kb`。第四周收集 Round-1 轨迹，执行 quarantine 写回和 promotion 门控，为后续 Round-2 演化实验准备 `KB v1`。