---
corpus_file: kbx_expanded_gap_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_sources.yaml
frozen_as: h20_expanded_pilot_run
ingested_at: 2026-07-09
updated_at: 2026-07-09
---

# KBX expanded pilot gap-check raw corpus notes

This cleaned corpus file records the gap check from the actual `experiments/h20/expanded_pilot_tasks.md` KernelBench-X task list back to existing raw corpus and KB coverage. It adds source-visible notes for old-family expanded variants that were not fully represented by the original MVP four-task corpus. It is not an H20 experiment result and contains no stable speedup claim.

## kbx_expanded_pilot_task_manifest — concrete task list source

- uri: experiments/h20/expanded_pilot_tasks.md
- kind: project_protocol; trust: project_report; license: repository
- topics: kernelbenchx, expanded_pilot, concrete_entrypoints, gap_check
- backends: nvidia_cuda; operators: pointwise, reduction, softmax, layernorm, matmul, layout_indexing
- candidate_fact_1: The expanded pilot freezes 36 concrete KBX-aligned tasks across six categories and lists the required top-level entrypoints.
- candidate_fact_2: KB coverage must match those concrete op names and entrypoints, not only the earlier abstract examples such as `row_softmax` or `layernorm_forward`.
- candidate_fact_3: This source is a manifest bridge; it does not include hidden tests, benchmark shapes, or performance measurements.
- applicability: all expanded pilot task generation and retrieval validation.
- limitations: use only for task mapping and coverage accounting; consult official operator docs and OpSpec for exact semantics.

## h20_mvp_benchmark_profile_protocol — repository benchmark/profile protocol

- uri: docs/h20_mvp_protocol.md
- kind: project_protocol; trust: project_report; license: repository
- topics: benchmark, profile, correctness_gate, p50, p95, memory_bound, stop_condition
- backends: nvidia_cuda; operators: common, pointwise, reduction, softmax, layernorm, matmul, layout_indexing
- candidate_fact_1: Performance testing is gated by correctness; benchmark/profile interpretation should happen after compile and correctness checks.
- candidate_fact_2: Repository benchmark summaries use repeated timing with p50/p95 style reporting rather than single-sample latency claims.
- candidate_fact_3: Profile summaries are normalized into coarse symptoms such as memory_bound, compute_bound, occupancy_limited, launch_bound, or unknown.
- candidate_fact_4: Stop-condition capsules may cite this protocol for the rule that repeated benchmark and profile attribution are required before concluding a tuning plateau.
- applicability: stop capsules and performance-candidate capsules that require benchmark_repeat and profile_attribution gates.
- limitations: this protocol source does not provide H20 performance numbers or prove a motif is faster; it only supports validation ordering and measurement requirements.

## pytorch_reduction_kbx_semantics — reduction variants beyond MVP sum/max

- uri:
  - https://docs.pytorch.org/docs/stable/generated/torch.std.html
  - https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
  - https://docs.pytorch.org/docs/stable/generated/torch.min.html
  - https://docs.pytorch.org/docs/stable/generated/torch.mean.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch stable docs accessed 2026-07-09
- topics: pytorch, reduction, std, mean, min, argmax, correction, keepdim, dtype
- backends: cpu, nvidia_cuda; operators: reduction
- candidate_fact_1: Expanded reduction tasks include `sum`, `mean`, `std`, `min`, `argmax`, and `fused_sum_std`, so old row-sum/max capsules are only partial coverage.
- candidate_fact_2: Reduction kernels must preserve `dim`, `keepdim`, optional dtype, and scalar-vs-tuple dimension behavior as declared by the task.
- candidate_fact_3: `std` carries a correction parameter and may need two-pass or numerically guarded statistics; `argmax` and `min(dim=...)` introduce index-output semantics in addition to value reduction.
- candidate_fact_4: `fused_sum_std` should share input reads only if it preserves both sum and std semantics; a performance fusion claim still requires H20 validation.
- applicability: `sum`, `mean`, `std`, `min`, `argmax`, `fused_sum_std` expanded tasks.
- limitations: no H20 timing or promotion evidence; derived KB entries remain candidate/quarantine unless validated.

## pytorch_softmax_ce_attention_semantics — fused softmax, CE, and attention variants

- uri:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch stable docs accessed 2026-07-09
- topics: pytorch, softmax, log_softmax, cross_entropy, attention, causal_mask, dtype
- backends: cpu, nvidia_cuda; operators: softmax, matmul
- candidate_fact_1: Expanded softmax tasks go beyond row softmax and include fused log-softmax, repeat-interleave + log-softmax, cross entropy + log-softmax, and attention.
- candidate_fact_2: Softmax/log-softmax subexpressions must preserve numerical stability and output dtype semantics before fusion or tiling choices.
- candidate_fact_3: Cross entropy introduces target indices, ignore_index, optional class weights, reduction modes, and label_smoothing; these are semantic conditions, not performance knobs.
- candidate_fact_4: Attention combines matmul-like score computation, optional causal masking, softmax, and value aggregation; it should not be treated as a pure row-softmax task.
- applicability: `softmax`, `fused_softmax_log`, `fused_log_softmax_linear`, `fused_repeat_interleave_log_softmax`, `fused_cross_entropy_log_softmax`, `attention`.
- limitations: no hidden-shape details or H20 flash-attention performance claim.

## pytorch_normalization_rms_dropout_semantics — normalization and fused normalization variants

- uri:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch stable docs accessed 2026-07-09
- topics: pytorch, layer_norm, RMSNorm, dropout, gelu, fused_normalization
- backends: cpu, nvidia_cuda; operators: layernorm, normalization
- candidate_fact_1: Expanded normalization tasks include W8A8 layernorm, fused layer_norm + ReLU + linear, cross entropy + softmax + layernorm, conv2d + layer_norm + SiLU, and bmm + RMSNorm + GELU + dropout variants.
- candidate_fact_2: Existing layernorm-forward capsules cover mean/variance/eps/gamma/beta basics but do not fully cover RMSNorm, dropout training/eval behavior, quantized W8A8 policy, or fused conv/bmm subgraphs.
- candidate_fact_3: RMSNorm normalizes by root mean square rather than subtracting the mean; do not reuse layernorm mean-subtraction code unless the OpSpec declares layernorm semantics.
- candidate_fact_4: Dropout in fused tasks must respect training/eval behavior and randomness policy defined by the harness; unverified deterministic shortcuts must remain invalid or candidate-only.
- applicability: expanded normalization/fused-normalization KBX tasks.
- limitations: no H20 fused-normalization benchmark claim; complex fused tasks may still require harness/task-specific legality support.

## pytorch_matmul_addmm_bmm_semantics — matmul variants and epilogues beyond MVP GEMM

- uri:
  - https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
  - https://docs.pytorch.org/docs/stable/generated/torch.bmm.html
  - https://docs.pytorch.org/docs/stable/generated/torch.addmm.html
  - https://docs.pytorch.org/docs/stable/generated/torch.tril.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch stable docs accessed 2026-07-09
- topics: pytorch, matmul, bmm, addmm, matvec, triangular_mask, epilogue
- backends: cpu, nvidia_cuda; operators: matmul
- candidate_fact_1: Expanded matmul tasks include generic matmul, fp16/bf16 matmul, addmm, matrix-vector dot, and triangular-mask matmul with scale.
- candidate_fact_2: Existing GEMM tile capsules cover the core contraction but do not fully cover addmm alpha/beta epilogue, matvec shape cases, batched matmul dispatch, or triangular mask semantics.
- candidate_fact_3: Eager cuBLAS/PyTorch matmul fallback remains illegal for kernel-authoring mode even when the task is matmul-like.
- candidate_fact_4: Epilogue fusion can be a candidate only after the base matmul semantics and boundary masks are hidden-correct.
- applicability: `matmul`, `matmul_fp16`, `matmul_bf16`, `addmm`, `matrix_vector_dot`, `tril_mm_and_scale`.
- limitations: no tensor-core tile-size or epilogue speedup claim without H20 measurements.

## Coverage status from the gap check

- Pointwise / fused memory-bound: supplemented by `pointwise_fusion_corpus.md` and KBX-bridge capsules; old MVP corpus did not cover concrete KBX `add`, `mul`, `gelu_fp16`, `gelu_bf16`, `fused_add_gelu`, or `fused_mul_sub` names.
- Reduction: base row reduction coverage exists for sum/max/tail masks/fp32 accumulation, but `std`, `argmax`, `min` value+index, tuple dims, and `fused_sum_std` need candidate capsules until H20 evidence exists.
- Softmax / attention: base row-softmax stability coverage exists, but fused cross entropy/log-softmax/repeat-interleave and attention need candidate capsules until exact task OpSpecs and H20 results validate them.
- Normalization: base layernorm-forward coverage exists, but RMSNorm/dropout/quantized W8A8/fused conv/bmm tasks are not fully covered by old LayerNorm capsules.
- Matmul-like: base GEMM/tl.dot/tile coverage exists, but addmm, matvec, triangular-mask epilogue, and bmm/fused-normalization subgraphs need additional candidate knowledge.
- Layout / indexing / irregular memory: supplemented by `layout_indexing_corpus.md` and concrete KBX capsules for index_select, permute_copy, scatter, masked_select, expand_where, and fused_gather_masked_fill.
