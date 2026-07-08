---
corpus_file: row_reduction_corpus
derived_from: sources/registry/sources.yaml
frozen_as: raw_corpus_v0
ingested_at: 2026-07-06
---
# row_reduction raw corpus notes
This cleaned Markdown corpus stores metadata, compact excerpts/locators, structured notes, applicability, limitations, candidate facts, and source links. It intentionally avoids long copyrighted source dumps.

## triton_sum_api — Triton tl.sum API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.sum.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, reduction; backends: nvidia_cuda, amd_rocm; operators: common, row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_max_api — Triton tl.max API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.max.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, reduction; backends: nvidia_cuda, amd_rocm; operators: common, row_reduction, softmax
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_sum — PyTorch torch.sum API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.sum.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_amax — PyTorch torch.amax API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.amax.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_max — PyTorch torch.max API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.max.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_mean — PyTorch torch.mean API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.mean.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_sum_reduction — Triton tl.sum reduction API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.sum.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_max_reduction — Triton tl.max reduction API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.max.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_reduction_sample — CUDA reduction sample
- uri: https://github.com/NVIDIA/cuda-samples
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_parallel_reduction_whitepaper — CUDA parallel reduction whitepaper
- uri: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cub_device_reduce — CUB DeviceReduce docs
- uri: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## thrust_reduce — Thrust reduce docs
- uri: https://nvidia.github.io/cccl/thrust/api/group__reductions.html
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## tritonbench_reduce — TritonBench reduction tasks
- uri: https://arxiv.org/abs/2502.14752
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_reduction_source — PyTorch reduction native source
- uri: https://github.com/pytorch/pytorch
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_warp_reductions — CUDA warp reductions
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## ncu_memory_reduction — Nsight memory metrics for reductions
- uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_masking_reduction — Triton masking for reductions
- uri: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## benchmark_row_reduce — Public row-reduction benchmark context
- uri: https://github.com/triton-lang/triton
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## inductor_reduction_codegen — TorchInductor reduction codegen context
- uri: https://github.com/pytorch/pytorch/tree/main/torch/_inductor
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_effective_bandwidth — CUDA effective bandwidth calculation
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effective-bandwidth-calculation
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## row_reduce_forum_failures — Community reduction failure modes
- uri: https://github.com/triton-lang/triton/issues
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, cpu; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_language_where — Triton tl.where API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.where.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: row_reduction, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: row_reduction
- candidate_fact_1: Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- candidate_fact_2: fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- candidate_fact_3: Tail masks are mandatory for non-power-of-two row lengths.
- candidate_fact_4: Memory bandwidth is often the primary limiter after correctness.
- candidate_fact_5: One-program-per-row is a starting skeleton, not a universal H20 optimum.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.
