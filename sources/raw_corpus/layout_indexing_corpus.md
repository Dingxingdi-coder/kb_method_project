---
corpus_file: layout_indexing_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_sources.yaml
frozen_as: h20_expanded_pilot_run
ingested_at: 2026-07-08
updated_at: 2026-07-09
---

# Layout / indexing / irregular memory raw corpus notes

This cleaned corpus file adds traceable source notes for the expanded pilot category "Layout / indexing / irregular memory". It is not an H20 benchmark result and contains no stable speedup claim.

## pytorch_stride_contiguous_tensor_views — PyTorch tensor views, strides, and contiguous

- uri:
  - https://docs.pytorch.org/docs/2.12/tensor_view.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.stride.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.contiguous.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, tensor view, stride, contiguous, non-contiguous layout
- backends: cpu, nvidia_cuda; operators: common, layout_indexing
- candidate_fact_1: A tensor view can share storage with a base tensor and may be non-contiguous.
- candidate_fact_2: A stride is the jump needed to move from one logical element to the next along a dimension.
- candidate_fact_3: `contiguous()` returns the same tensor if already in the target memory format and otherwise returns a contiguous copy.
- candidate_fact_4: A custom layout/indexing kernel should compute storage offsets from logical indices and strides rather than assume `base + linear_offset`.
- candidate_fact_5: In kernel-authoring mode, calling `contiguous()`, `permute()`, or `reshape()` is review-required and can be illegal if it performs the target copy/layout transform.
- applicability: transpose/permute copy, slice+concat, strided copy, layout transform, and non-contiguous operands in gather/scatter tasks.
- limitations: semantic source only; no H20 performance evidence.

## pytorch_gather_index_select_semantics — PyTorch gather and index_select semantics

- uri:
  - https://docs.pytorch.org/docs/2.12/generated/torch.gather.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.index_select.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, gather, index_select, embedding lookup, index dtype, bounds
- backends: cpu, nvidia_cuda; operators: layout_indexing
- candidate_fact_1: `torch.gather` gathers values along a declared dimension; input and index have the same number of dimensions and do not broadcast against each other.
- candidate_fact_2: Gather output shape follows the index shape.
- candidate_fact_3: The index tensor for gather is a LongTensor in PyTorch documentation.
- candidate_fact_4: `index_select` selects along a dimension with an int32 or int64 1-D index tensor and returns a new tensor.
- candidate_fact_5: Custom gather/embedding kernels must check index bounds and use the task-declared index dtype; do not infer sorted, unique, or contiguous indices unless OpSpec guarantees them.
- applicability: gather rows, embedding lookup, indexed row copy, irregular read-side kernels.
- limitations: does not specify a universal high-performance schedule for arbitrary index distributions.

## pytorch_scatter_add_semantics — PyTorch scatter_add and scatter semantics

- uri:
  - https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.scatter_add_.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.scatter_.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, scatter, scatter_add, atomics, collision, nondeterminism, index dtype
- backends: cpu, nvidia_cuda; operators: layout_indexing
- candidate_fact_1: `scatter_add_` adds source values into destination positions specified by an index tensor.
- candidate_fact_2: `self`, `index`, and `src` must have the same number of dimensions; index/src do not broadcast in scatter_add.
- candidate_fact_3: PyTorch documents CUDA nondeterminism for scatter_add and only implements backward for `src.shape == index.shape`.
- candidate_fact_4: Scatter-style duplicate indices are a collision case; scatter-add must accumulate all contributions rather than use a plain overwrite store.
- candidate_fact_5: A correct custom scatter-add normally requires atomic add or an equivalent staged conflict-resolution strategy unless OpSpec guarantees unique indices.
- applicability: scatter-add, embedding backward-like accumulation, irregular write-side kernels.
- limitations: this source supports semantic and validation rules; performance depends on collision distribution and backend atomic behavior.

## pytorch_masked_select_where_semantics — PyTorch masked_select and where semantics

- uri:
  - https://docs.pytorch.org/docs/stable/generated/torch.masked_select.html
  - https://docs.pytorch.org/docs/stable/generated/torch.where.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch stable docs accessed 2026-07-09
- topics: pytorch, masked_select, boolean mask, where, broadcasting, output shape
- backends: cpu, nvidia_cuda; operators: layout_indexing, pointwise
- candidate_fact_1: `masked_select(input, mask)` selects elements where a Boolean mask is true; input and mask are broadcastable and the result is a new one-dimensional tensor.
- candidate_fact_2: `where(condition, input, other)` returns elements selected from `input` or `other`; condition, input, and other must be broadcastable.
- candidate_fact_3: A custom `masked_select` kernel must implement compaction/prefix-count semantics, not merely write masked positions into the original dense shape.
- candidate_fact_4: A custom `expand_where` kernel must preserve expand/broadcast address rules and produce the broadcasted output shape.
- applicability: `masked_select`, `expand_where`, and fused masked-fill/where-like layout tasks.
- limitations: these are semantic rules only; no H20 compaction or where performance claim is included.

## pytorch_scatter_overwrite_semantics — PyTorch scatter overwrite semantics

- uri: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch stable docs accessed 2026-07-09
- topics: pytorch, scatter, index bounds, duplicate indices, nondeterminism, overwrite semantics
- backends: cpu, nvidia_cuda; operators: layout_indexing
- candidate_fact_1: Scatter writes source values into positions declared by an index tensor along a dimension; index values must be within output bounds.
- candidate_fact_2: PyTorch warns that non-unique indices can make scatter behavior nondeterministic and can propagate gradients incorrectly.
- candidate_fact_3: Scatter overwrite differs from scatter-add: duplicate indices do not mean all contributions are summed.
- candidate_fact_4: A custom scatter kernel must follow the task's duplicate-index policy and must not silently substitute scatter-add or unique-index assumptions.
- applicability: KBX `scatter` and other indexed write-side tasks.
- limitations: this source supports legality/correctness boundaries, not an H20 atomic or overwrite performance conclusion.

## cuda_best_practices_memory_coalescing — CUDA Best Practices Guide

- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- kind: official_doc; trust: official; license: unknown; version/date: CUDA 13.3 docs
- topics: cuda, coalescing, strided access, shared memory tile, bank conflict, effective bandwidth
- backends: nvidia_cuda; operators: common, layout_indexing
- candidate_fact_1: Coalesced global memory access is preferred because adjacent lanes accessing adjacent words need fewer memory transactions.
- candidate_fact_2: Strided accesses waste bandwidth because fetched memory segments can contain unused words.
- candidate_fact_3: Regular transpose/permute copies can use tiled schedules to make one side of the copy coalesced; shared-memory tiles can help but can create bank conflicts or occupancy pressure.
- candidate_fact_4: Irregular gather/scatter indices can defeat coalescing; treat index loads, source reads, and destination writes as separate memory legs.
- candidate_fact_5: Effective bandwidth and profiler throughput metrics should be used after correctness, not as standalone evidence of correctness or stable H20 speedup.
- applicability: transpose/permute copy, strided copy, layout transform, gather rows, embedding lookup, scatter-add.
- limitations: CUDA guide examples are general; H20-specific tile sizes, atomic behavior, and coalescing outcomes require local benchmarks.

## triton_vector_add_pointwise_tail_mask — Triton vector add tutorial

- uri: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, program_id, tl.arange, masks, load/store
- backends: nvidia_cuda, amd_rocm; operators: common, layout_indexing
- candidate_fact_1: The same `tl.program_id` + `tl.arange` + mask pattern used for contiguous vector add is a reusable skeleton for tail-safe copy/index kernels.
- candidate_fact_2: For layout kernels, replace contiguous offsets with stride-aware or index-derived offsets while preserving masks for out-of-bounds lanes.
- applicability: strided copy, transpose/permute edge tiles, gather/scatter bounds masks.
- limitations: vector add is not an irregular memory benchmark.

## kbx_expanded_pilot_layout_entrypoints — concrete KBX task bridge

- uri: experiments/h20/expanded_pilot_tasks.md
- kind: project_protocol; trust: project_report; license: repository
- topics: kernelbenchx, layout_indexing, concrete entrypoints, irregular memory
- backends: nvidia_cuda; operators: layout_indexing
- candidate_fact_1: The expanded pilot concrete Layout / Indexing / Irregular Memory tasks are `index_select`, `permute_copy`, `scatter`, `masked_select`, `expand_where`, and `fused_gather_masked_fill`.
- candidate_fact_2: These KBX op names need retrieval coverage from stride-aware indexing, gather/index bounds, scatter duplicate-index policy, boolean-mask compaction, broadcasted where, and irregular coalescing capsules.
- candidate_fact_3: `permute_copy` is a regular materialized layout transform candidate; `index_select` and `fused_gather_masked_fill` are indexed read-side tasks; `scatter` is indexed write-side; `masked_select` is compacting mask selection; `expand_where` combines broadcasted view semantics and conditional pointwise selection.
- candidate_fact_4: This entry is a task-manifest bridge only; it does not expose hidden tests, benchmark shapes, or H20 performance measurements.
- applicability: actual KBX-expanded layout/indexing tasks generated from `experiments/h20/expanded_pilot_tasks.md`.
- limitations: entrypoint mapping does not prove any schedule is faster on H20.

## Candidate task mapping

- transpose / permute copy: compute source and destination logical coordinates explicitly; use tile masks at edges; consider tiled coalesced copy only after correctness.
- gather rows: validate index dtype and bounds; do not assume sorted or unique indices; coalesce within row payload if row data is contiguous.
- embedding lookup: treat as gather rows with row-wise payload; separate index loads from row data loads.
- scatter-add: preserve additive collision semantics with atomics or staged reduction; validate duplicate indices and CUDA nondeterminism tolerance.
- slice + concat: partition output logical space into source regions; apply source base offsets and bounds.
- strided copy / layout transform: compute every source and destination storage offset from strides; do not use high-level `contiguous()` or `permute()` to perform the target copy in the submitted path.
- KBX `index_select`: use gather/index-select semantics with declared dim, int32/int64 index support as required by task, and bounds validation.
- KBX `permute_copy`: materialize the permuted output; do not rely on a view-only permute or Python-level `contiguous()` fallback.
- KBX `scatter`: implement overwrite semantics and duplicate-index policy from OpSpec; do not replace it with scatter-add.
- KBX `masked_select`: implement Boolean mask broadcasting and one-dimensional compaction semantics.
- KBX `expand_where`: preserve expand/broadcast zero-stride semantics and conditional selection in the low-level kernel.
- KBX `fused_gather_masked_fill`: combine gather/index addressing and mask/value replacement without delegating either target suboperation to high-level PyTorch.
