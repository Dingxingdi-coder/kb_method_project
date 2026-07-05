# H20 自我进化闭环实验

## 1. Round-0: 初始资料库与初始知识库

Round-0 先构建并冻结初始资料库，再从同一资料库转换出初始知识库。

输入来源：

- 本项目调研报告抽取的通用方法知识；
- Triton/PyTorch/CUDA 官方文档；
- 自动算子生成、kernel 优化、correctness harness、profiling 和 autotune 相关论文；
- 5–10 个已知正确的基础 skeleton；
- 示例 kernel、benchmark 说明和公开工具链资料；
- 0 个 H20 性能经验，或只保留 manual baseline 的非 promoted 证据。

输出：

```text
raw_corpus_v0
KB v0
raw_corpus_rag_index_v0
kb_plain_rag_index_v0
ecc_kb_index_v0
```

`KB v0` 必须只从 `raw_corpus_v0` 抽取。A1、A2、A3 的资料来源因此保持一致。

## 2. Round-1: 四组主实验

对 `operators.md` 中公开 shape 跑四组 baseline：

```text
A0_prompt
A1_raw_corpus_rag
A2_kb_plain_rag
A3_ecc_kb
```

A1 使用 `raw_corpus_rag_index_v0`。A2 使用 `kb_plain_rag_index_v0`。A3 使用 `ecc_kb_index_v0` 和 `KB v0` 的 ContextPacket 服务。

每次 run 后生成 trace record：

```text
op_spec_hash
backend_fingerprint
source_corpus_version
kb_version
context_packet_hash_or_rag_context_hash
candidate_hash
compile/correctness/performance/profile result
decision KEEP/DISCARD/FAIL
retrieved_item_ids
used_capsule_ids if applicable
token/time/iteration cost
```

A0、A1、A2 的 trace 只用于评估和审计。A3 的 trace 额外进入 ECC-KB 自我进化管道。

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

## 4. 后续 Round-2: Held-out 验证

Round-2 不属于首轮四组实验的必做范围，但后续必须执行。Round-2 用 held-out shape/dtype/算子变体测试 `KB v1`。禁止使用 Round-2 的结果更新知识后再评估同一 Round-2 任务。

后续比较：

```text
A3(v0 frozen) vs A3(v1 promoted)
A3(v1) vs A0/A1/A2
A2(v1) if budget allows
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