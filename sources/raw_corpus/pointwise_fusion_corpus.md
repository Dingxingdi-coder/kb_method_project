---
corpus_file: pointwise_fusion_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_sources.yaml
frozen_as: h20_expanded_pilot_run
ingested_at: 2026-07-08
updated_at: 2026-07-09
---

# Pointwise / fused memory-bound raw corpus notes

This cleaned corpus file adds traceable source notes for the expanded pilot category "Pointwise / fused memory-bound". It is not an H20 benchmark result and contains no stable speedup claim.

## triton_vector_add_pointwise_tail_mask — Triton vector add tutorial

- uri: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, pointwise, fused elementwise, tail mask, tl.load, tl.store, benchmark
- backends: nvidia_cuda, amd_rocm; operators: common, pointwise
- candidate_fact_1: A simple Triton pointwise kernel maps `tl.program_id(axis=0)` to a contiguous block of element offsets.
- candidate_fact_2: The tutorial guards out-of-bounds tail elements with `mask = offsets < n_elements` and applies the same mask to `tl.load` and `tl.store`.
- candidate_fact_3: The same skeleton can be adapted from vector add to fused pointwise expressions such as bias+GELU, bias+SiLU, residual+activation, broadcast affine, gated multiply, and clamp/mul/add.
- candidate_fact_4: Correctness should be checked against a reference before benchmark or profile-driven tuning.
- applicability: legal only when the submitted path implements the target expression in the custom kernel and does not delegate target semantics to PyTorch high-level fallback.
- limitations: the tutorial demonstrates a basic contiguous vector add pattern, not an H20-verified fused activation speedup.

## cuda_best_practices_memory_coalescing — CUDA Best Practices Guide

- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- kind: official_doc; trust: official; license: unknown; version/date: CUDA 13.3 docs
- topics: cuda, global memory, coalescing, effective bandwidth, memory-bound profiling
- backends: nvidia_cuda; operators: common, pointwise, layout_indexing
- candidate_fact_1: Effective bandwidth is computed from bytes read plus bytes written divided by kernel time; this can be used as a profile-derived symptom, not as a standalone success claim.
- candidate_fact_2: Coalesced global-memory accesses reduce wasted memory transactions for adjacent lane accesses; scattered or strided accesses can waste bandwidth.
- candidate_fact_3: For fused pointwise kernels with contiguous main operands, prefer contiguous lane-to-element mapping for loads and stores.
- candidate_fact_4: Stop local tuning only after hidden correctness, repeated benchmarks, and profile attribution show a memory-bound plateau; do not infer an H20 conclusion from documentation alone.
- applicability: use as candidate guidance for memory-bound pointwise tasks after correctness gates pass.
- limitations: CUDA guide examples are not H20 pilot measurements; concrete block sizes and vector widths require local validation.

## pytorch_broadcasting_expand_semantics — PyTorch broadcasting and expand semantics

- uri:
  - https://docs.pytorch.org/docs/2.12/notes/broadcasting.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.expand.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, broadcasting, expand, stride zero, elementwise semantics
- backends: cpu, nvidia_cuda; operators: common, pointwise, layout_indexing
- candidate_fact_1: Broadcastable tensors are aligned from trailing dimensions; each aligned dimension must be equal, one, or absent.
- candidate_fact_2: The resulting broadcasted shape uses the maximum size across aligned dimensions after prepending leading ones as needed.
- candidate_fact_3: Expanding a singleton dimension can be represented as a view with stride 0 rather than a data copy.
- candidate_fact_4: For custom fused pointwise kernels, compute each operand address using its logical shape/stride and use zero offset along broadcasted singleton dimensions.
- candidate_fact_5: In-place writes to expanded/broadcasted aliases are dangerous because multiple logical elements may refer to the same memory location.
- applicability: pointwise broadcast affine, bias+activation, gated multiply, residual+activation, clamp/mul/add, and layout kernels with expanded operands.
- limitations: this is semantic documentation, not performance evidence.

## pytorch_gelu_silu_semantics — PyTorch GELU and SiLU formulas

- uri:
  - https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.gelu.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.silu.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, activation, gelu, silu, numerical semantics
- backends: cpu, nvidia_cuda; operators: pointwise
- candidate_fact_1: GELU has an exact definition and an optional tanh approximation; the OpSpec must say which is the reference.
- candidate_fact_2: SiLU is elementwise `x * sigmoid(x)`.
- candidate_fact_3: For fp16/bf16 inputs, an fp32 intermediate for activation math is a conservative correctness choice when tolerances require it.
- candidate_fact_4: Calling `torch.nn.functional.gelu`, `torch.nn.functional.silu`, or equivalent high-level PyTorch operations in the submitted target path is a legality risk under kernel-authoring mode.
- applicability: bias+GELU, bias+SiLU, residual+activation, gated multiply `silu(x1) * x2`.
- limitations: formula support does not prove which approximation or dtype is fastest on H20.

## h20_formal_kernel_authoring_rules — repository kernel-authoring protocol

- uri: docs/h20_formal_experiment_protocol.md
- kind: project_protocol; trust: project_report; license: repository
- topics: legality, anti-cheating, kernel-authoring mode, pointwise fallback
- backends: nvidia_cuda; operators: common, pointwise, layout_indexing
- candidate_fact_1: The formal protocol evaluates custom kernel authoring, not high-level library composition.
- candidate_fact_2: Output allocation and metadata reads are allowed infrastructure actions, but target semantics must not be delegated to PyTorch/ATen/cuDNN/cuBLAS or other high-level fallback.
- candidate_fact_3: Review-required APIs such as `contiguous()`, `reshape()`, `permute()`, and `as_strided()` become illegal if they perform the target data movement or computation.
- applicability: all expanded pilot kernel-authoring tasks.
- limitations: this is a repository legality rule, not an upstream CUDA or PyTorch performance rule.

## kbx_expanded_pilot_pointwise_entrypoints — concrete KBX task bridge

- uri: experiments/h20/expanded_pilot_tasks.md
- kind: project_protocol; trust: project_report; license: repository
- topics: kernelbenchx, pointwise, fused memory-bound, concrete entrypoints, task bridge
- backends: nvidia_cuda; operators: pointwise
- candidate_fact_1: The expanded pilot concrete Pointwise / Fused Memory-Bound tasks are `add`, `mul`, `gelu_fp16`, `gelu_bf16`, `fused_add_gelu`, and `fused_mul_sub`.
- candidate_fact_2: These concrete KBX op names must retrieve the same pointwise correctness, legality, tail-mask, broadcast-indexing, activation-intermediate, and memory-bound candidate knowledge as the abstract design examples.
- candidate_fact_3: `add` and `mul` are simpler elementwise forms; `gelu_fp16`, `gelu_bf16`, and `fused_add_gelu` need activation formula and dtype/tolerance checks; `fused_mul_sub` needs fused arithmetic without materializing temporary tensors.
- candidate_fact_4: This entry is a task-manifest bridge only; it does not add benchmark shapes, hidden-test details, or H20 performance measurements.
- applicability: actual KBX-expanded pointwise tasks generated from `experiments/h20/expanded_pilot_tasks.md`.
- limitations: entrypoint mapping does not prove a fused custom kernel is faster than any baseline.

## Candidate task mapping

- bias + GELU: use broadcast address calculation, GELU formula, optional fp32 intermediate, tail masks, and no high-level activation fallback.
- bias + SiLU: use broadcast address calculation, SiLU formula, optional fp32 intermediate, tail masks, and no high-level activation fallback.
- residual add + activation: compute residual and activation in one kernel if legal and hidden-correct; avoid temporary tensors.
- broadcast affine `y = x * scale + bias`: compute `x`, `scale`, and `bias` addresses from output logical indices and per-operand broadcast/stride metadata.
- gated multiply `y = silu(x1) * x2`: preserve SiLU formula and gated multiplication order; cast intermediates according to OpSpec tolerance.
- clamp + mul + add: implement clamp bounds and arithmetic in the low-level kernel; keep output dtype per OpSpec.
- KBX `add` / `mul`: use the same tail-safe pointwise skeleton and honor scalar/tensor operand broadcasting plus optional alpha/out semantics declared by the task.
- KBX `gelu_fp16` / `gelu_bf16`: preserve the `approximate` mode and validate activation precision before performance tuning.
- KBX `fused_add_gelu`: fuse add and GELU in the custom kernel; do not call PyTorch add/GELU to implement the target semantic path.
- KBX `fused_mul_sub`: fuse multiply and subtract in one low-level path; validate broadcasting, tail masks, and output dtype.
