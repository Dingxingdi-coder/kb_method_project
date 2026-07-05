# H20 MVP 指标体系

## 1. Correctness 指标

```text
compile_success_rate
smoke_correctness_pass_rate
quick_correctness_pass_rate
hidden_correctness_pass_rate
robustness_pass_rate
false_positive_rate_if_rechecked
cheating_detected_count
```

正确性优先级高于性能。未通过 hidden correctness 的候选不得进入性能比较。

## 2. Performance 指标

```text
latency_p50_ms
latency_p95_ms
latency_mean_ms
latency_std_ms
speedup_vs_eager_p50
speedup_vs_torch_compile_p50
correct_and_faster_rate_vs_eager
correct_and_faster_rate_vs_torch_compile
bandwidth_or_tflops_if_applicable
profile_symptom_distribution
```

性能必须报告 baseline 类型，禁止只说“更快”。

## 3. Efficiency 指标

```text
iterations_to_first_compile
iterations_to_first_correct
iterations_to_target_speedup
token_to_first_correct
token_to_target_speedup
wall_time_to_first_correct
wall_time_to_target_speedup
gpu_benchmark_runs_per_task
invalid_compile_attempts_per_task
```

这些指标直接对应“少走弯路”。

## 4. Context and Knowledge Utility 指标

四组都必须记录上下文规模。A1、A2、A3 还必须记录召回来源，方便区分“资料有用”“知识表示有用”和“ECC-KB 方法有用”。

```text
context_token_count
retrieved_item_count
retrieved_item_ids
retrieved_source_type_distribution
retrieved_context_used_by_agent_rate
```

A1 专属指标：

```text
raw_corpus_retrieved_chunk_count
raw_corpus_context_tokens
raw_corpus_chunk_adoption_rate
raw_corpus_chunk_effective_rate
```

A2 专属指标：

```text
kb_plain_rag_retrieved_unit_count
kb_plain_rag_context_tokens
kb_plain_rag_unit_adoption_rate
kb_plain_rag_unit_effective_rate
```

A3 专属指标：

```text
retrieved_capsule_count
context_packet_tokens
capsule_adoption_rate
capsule_effective_rate
anti_action_prevented_count
failure_signature_hit_rate
stop_capsule_saved_attempts
promoted_capsule_reuse_rate_round2
stale_or_wrong_capsule_hit_rate
```

`*_effective_rate` 定义为：被 Agent 明确采用或由 harness 触发后，对 compile/correctness/performance/iteration 至少一项带来可观改善的资料片段、知识单元或 capsule 比例。

## 5. Evolution 指标

```text
quarantine_records_count
promoted_records_count
rejected_records_count
promotion_precision
round2_delta_correctness
round2_delta_speedup
round2_delta_token
round2_delta_iterations
heldout_generalization_gain
```

知识库进化必须在 held-out tasks 上体现，而不能只在原任务复用。Round-1 首轮四组实验可以先只收集 evolution 原始记录，不要求完成 Round-2。

## 6. 最小统计方式

每个任务每组至少 3 次独立 run。汇总时报告 median 和 bootstrap confidence interval。如果预算不足，先报告 paired task-level comparison，而不是只报告整体平均。

A0、A1、A2、A3 必须在同一任务集合、同一 agent、同一 token 上限、同一 wall-clock 上限和同一 GPU benchmark 预算下比较。A1、A2、A3 的上下文 token 上限必须相同。