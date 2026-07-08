---
corpus_file: matmul_corpus
derived_from: sources/registry/sources.yaml
frozen_as: raw_corpus_v0
ingested_at: 2026-07-06
---
# matmul raw corpus notes
This cleaned Markdown corpus stores metadata, compact excerpts/locators, structured notes, applicability, limitations, candidate facts, and source links. It intentionally avoids long copyrighted source dumps.

## triton_matmul_tutorial — Triton matmul tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, matmul, tiling; backends: nvidia_cuda, amd_rocm; operators: common, matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_persistent_matmul — Triton persistent matmul tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, persistent_matmul; backends: nvidia_cuda, amd_rocm; operators: common, matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_dot_api — Triton tl.dot API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, dot, matmul; backends: nvidia_cuda, amd_rocm; operators: common, matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_matmul — PyTorch torch.matmul API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.matmul.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_mm — PyTorch torch.mm API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.mm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_bmm — PyTorch torch.bmm API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.bmm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_addmm — PyTorch torch.addmm API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_matmul_tutorial_detail — Triton matmul tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_matmul_source — Triton matmul example source
- uri: https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_dot_matmul — Triton tl.dot for matmul
- uri: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_persistent_matmul_detail — Triton persistent matmul tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_group_gemm — Triton group GEMM tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_block_scaled_matmul — Triton block scaled matmul tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cutlass_gemm_docs — CUTLASS GEMM docs
- uri: https://github.com/NVIDIA/cutlass/tree/main/media/docs
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cutlass_gemm_examples — CUTLASS GEMM examples
- uri: https://github.com/NVIDIA/cutlass/tree/main/examples
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cublas_gemm_docs — cuBLAS GEMM docs
- uri: https://docs.nvidia.com/cuda/cublas/index.html
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_tensor_core — CUDA tensor core programming
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_tf32_cuda — PyTorch TF32 CUDA note
- uri: https://docs.pytorch.org/docs/2.12/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_matmul_precision — PyTorch set_float32_matmul_precision API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.set_float32_matmul_precision.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_shared_matmul — CUDA shared-memory matmul tiling
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_coalescing_matmul — CUDA coalesced access for matmul
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## splitk_gemm_paper — Split-K GEMM paper
- uri: https://arxiv.org/abs/2402.00025
- kind: paper; trust: peer_reviewed; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## ncu_matmul_compute — Nsight compute-bound context for GEMM
- uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: matmul, h20_mvp; backends: nvidia_cuda, cpu; operators: matmul
- candidate_fact_1: Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- candidate_fact_2: Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- candidate_fact_3: Boundary masks are required for non-multiple tile sizes.
- candidate_fact_4: TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- candidate_fact_5: Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.
