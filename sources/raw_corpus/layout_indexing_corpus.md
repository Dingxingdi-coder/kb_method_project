---
corpus_file: layout_indexing_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_sources.yaml
frozen_as: h20_expanded_pilot_seed
ingested_at: 2026-07-08
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

## Candidate task mapping

- transpose / permute copy: compute source and destination logical coordinates explicitly; use tile masks at edges; consider tiled coalesced copy only after correctness.
- gather rows: validate index dtype and bounds; do not assume sorted or unique indices; coalesce within row payload if row data is contiguous.
- embedding lookup: treat as gather rows with row-wise payload; separate index loads from row data loads.
- scatter-add: preserve additive collision semantics with atomics or staged reduction; validate duplicate indices and CUDA nondeterminism tolerance.
- slice + concat: partition output logical space into source regions; apply source base offsets and bounds.
- strided copy / layout transform: compute every source and destination storage offset from strides; do not use high-level `contiguous()` or `permute()` to perform the target copy in the submitted path.
