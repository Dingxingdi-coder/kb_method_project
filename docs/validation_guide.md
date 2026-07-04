# 读者验收指南

## 1. 先读哪些内容

判断方向是否正确，先读：

1. `docs/method.md` 的知识定义、双层结构、自我进化机制；
2. `schemas/knowledge_unit.schema.json`，确认知识不是文档 chunk；
3. `docs/h20_mvp_protocol.md`，确认只用 H20 就能验证；
4. `experiments/h20/baselines.md` 和 `metrics.md`，确认有可比较 baseline；
5. `examples/operator_walkthrough.md`，确认机制能落到一个算子上。

## 2. 什么现象说明方案退化为普通 RAG

出现以下情况，说明实现偏离 ECC-KB：

- 只把论文、文档和代码切 chunk 后向量检索；
- ContextPacket 中没有 must_obey、anti_actions、validation_plan 和 stop_conditions；
- 检索不感知 phase、op spec、shape/dtype/layout、backend fingerprint；
- 成功经验没有 evidence 和 gate，失败经验只是日志；
- 知识没有 quarantine/promotion/stale/reject 状态；
- 性能结论没有 p50/p95、版本和 correctness matrix；
- Agent 可以修改测试或放宽门槛。

## 3. 必须逐项检查的硬约束

- 后端芯片无关：是否先有 abstract backend model，再有 H20 instance。
- Coding Agent 无关：是否通过文件/API/CLI 协议接入，而不是依赖某 Agent 内存。
- 不依赖多后端硬件：H20 MVP 是否完整。
- 不训练大模型：第一阶段是否只用资料、代码、轨迹和真机反馈。
- 必须可验证：是否有 A0/A1/A2/A3 baseline 和固定指标。
- 主方案唯一：是否收敛到 Evidence Capsule + Evolution Gate，而不是多个候选方向。

## 4. H20 MVP 中最关键指标

最能证明 ECC-KB 有效的指标不是单个最快 kernel，而是：

- hidden correctness pass rate；
- iterations/token/time to first correct；
- correct-and-faster rate vs `torch.compile`；
- invalid compile attempts per task；
- repeated failure avoided count；
- retrieved capsule adoption rate；
- promoted capsule effective rate in Round-2 held-out tasks；
- p95 latency 不退化。

## 5. 什么结果说明方法不值得继续

出现以下结果，应考虑停止或重构：

- A3 相对 A1 普通 RAG 没有降低 token/time/iterations；
- A3 正确率没有提升，且上下文更长导致成本上升；
- promoted capsule 在 Round-2 无法复用，effective rate 接近随机；
- 失败记录不能阻止重复错误；
- 性能提升主要来自更多 benchmark 尝试，而不是知识缩小搜索空间。

## 6. 第一批实现文件

落地时先创建并实现：

```text
schemas/knowledge_unit.schema.json
schemas/evolution_record.schema.json
tools/validate_schema.py
tools/retrieve_context.py
tools/ingest_run.py
tools/promote_capsule.py
experiments/h20/operators.md
experiments/h20/baselines.md
experiments/h20/metrics.md
```

第一版工具可以很简单。关键是数据模型和门控必须正确。
