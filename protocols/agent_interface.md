# Coding Agent 无关接入协议

## 1. 文件协议

Agent 只需要读写以下文件：

```text
task.json
context_packet.json
candidate.py or candidate.cpp
notes.md
```

Harness 生成：

```text
results.json
compile.log
correctness.log
benchmark.json
profile_summary.json
trace.jsonl
```

Agent 不得修改 harness、reference、hidden tests、metrics definitions。

## 2. ContextPacket

ContextPacket 是知识库对 Agent 的唯一输出，不依赖 Agent 内部记忆：

```json
{
  "phase": "performance_optimize",
  "must_obey": [],
  "recommended_actions": [],
  "anti_actions": [],
  "validation_plan": [],
  "stop_conditions": [],
  "evidence_refs": []
}
```

## 3. CLI 草案

```bash
kbctl retrieve \
  --task task.json \
  --phase generate \
  --backend backend.json \
  --out context_packet.json

kbctl ingest-run \
  --trace trace.jsonl \
  --results results.json \
  --out kb/quarantine/run_001.json

kbctl promote \
  --record kb/quarantine/run_001.json \
  --schema schemas/evolution_record.schema.json
```

CLI 可替换为 API、MCP tool call 或文件系统轮询；协议字段保持不变。
