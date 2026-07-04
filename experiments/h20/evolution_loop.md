# H20 自我进化闭环实验

## 1. Round-0: 初始知识库

输入来源：

- 本项目调研报告抽取的通用方法知识；
- Triton/PyTorch 官方文档中人工结构化的少量合法性知识；
- 5–10 个已知正确的基础 skeleton；
- 0 个 H20 性能经验，或只保留 manual baseline 的非 promoted 证据。

输出：`KB v0`。

## 2. Round-1: 初始任务执行

对 `operators.md` 中公开 shape 跑四组 baseline。A3 使用 `KB v0`。

每次 run 后生成 evolution record：

```text
op_spec_hash
backend_fingerprint
context_packet_hash
candidate_hash
compile/correctness/performance/profile result
decision KEEP/DISCARD/FAIL
used_capsule_ids
token/time/iteration cost
```

## 3. Ingest and Promotion

A3 的所有轨迹先进入 quarantine。

Promotion gate：

```text
semantic capsule:
  must pass schema + source + hidden correctness relevance
failure capsule:
  needs reproducible signature or clear diagnostic value
performance capsule:
  correctness hidden pass
  repeated benchmark pass
  p50 improves or avoids known failure
  p95 no material regression
  valid_when/invalid_when filled
  profile or ablation supports mechanism
```

输出：`KB v1`。

## 4. Round-2: Held-out 验证

用 held-out shape/dtype/算子变体测试 `KB v1`。禁止使用 Round-2 的结果更新知识后再评估同一 Round-2 任务。

比较：

```text
A3(v0) vs A3(v1)
A3(v1) vs A0/A1/A2
```

## 5. 判断进化有效的条件

进化有效必须满足至少三条：

- `iterations_to_first_correct` 下降；
- `token_to_target_speedup` 下降；
- hidden correctness pass rate 上升；
- correct-and-faster rate 上升；
- failure_signature_hit 后同类失败减少；
- promoted capsule 在 held-out tasks 上 effective rate 明显高于未 promoted quarantine capsule；
- stale_or_wrong_capsule_hit_rate 低。

## 6. 反证条件

以下结果说明自我进化没有成立：

- v1 只增加 context length，token 更高而指标不变；
- promoted capsule 多数未被采用或采用后无效；
- 失败知识不能阻止重复错误；
- Round-1 有提升但 Round-2 无提升；
- performance capsule 导致 correctness 退化。
