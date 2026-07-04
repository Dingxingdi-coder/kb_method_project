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

先读 `docs/method.md` 判断方法是否有新意。再读 `docs/h20_mvp_protocol.md` 判断能否在 H20 上验证。然后读 `schemas/knowledge_unit.schema.json` 和 `schemas/evolution_record.schema.json` 判断知识是否可机器消费。最后读 `experiments/h20/*` 和 `examples/operator_walkthrough.md` 判断实验是否可落地。

## 最小落地路径

第一周实现 schema 校验、run 记录和人工填充的 20–40 个初始知识单元。第二周实现 H20 harness 和四组 baseline。第三周完成 Round-1 实验、quarantine 写回和 promotion 门控。第四周用 Round-2 的 held-out shape/dtype/算子变体验证自我进化是否有效。
