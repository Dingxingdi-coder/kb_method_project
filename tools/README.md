# Tools 草案

本目录第一阶段只需要轻量脚本，不需要完整平台。

建议最小工具：

```text
validate_schema.py      # 校验 knowledge_unit 和 evolution_record
retrieve_context.py     # 从 stable/quarantine/failures 读取并生成 ContextPacket
ingest_run.py           # 把 harness 结果压缩为 evolution_record
promote_capsule.py      # 执行 promotion gate
summarize_trace.py      # 将 Agent 轨迹压缩为 state-action-result-diagnosis
```

工具原则：

- 所有门控由工具执行，不让 Agent 自行决定是否通过；
- 所有结果写 JSON/JSONL，便于审计；
- 第一版可以不接向量库，先实现结构化过滤和人工 top-k；
- 检索增强是后续优化，不是方法核心。
