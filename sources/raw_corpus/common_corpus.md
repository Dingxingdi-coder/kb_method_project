---
corpus_file: common_corpus
derived_from: sources/registry/sources.yaml
frozen_as: raw_corpus_v0
ingested_at: 2026-07-06
---
# common raw corpus notes
This cleaned Markdown corpus stores metadata, compact excerpts/locators, structured notes, applicability, limitations, candidate facts, and source links. It intentionally avoids long copyrighted source dumps.

## triton_docs — Triton documentation index
- uri: https://triton-lang.org/main/index.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, dsl, programming_model; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_vector_add — Triton vector add tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, masking, program_id; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_jit_api — Triton jit API
- uri: https://triton-lang.org/main/python-api/generated/triton.jit.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, jit, constexpr; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_autotune_api — Triton autotune API
- uri: https://triton-lang.org/main/python-api/generated/triton.autotune.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, autotune; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_heuristics_api — Triton heuristics API
- uri: https://triton-lang.org/main/python-api/generated/triton.heuristics.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, heuristics; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_arange_api — Triton tl.arange API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.arange.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, indexing; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_load_api — Triton tl.load API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, load, mask; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_store_api — Triton tl.store API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.store.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, store, mask; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_program_id_api — Triton tl.program_id API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.program_id.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, grid; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_do_bench — Triton do_bench API
- uri: https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: benchmark, triton; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_perf_report — Triton perf_report API
- uri: https://triton-lang.org/main/python-api/generated/triton.testing.perf_report.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: benchmark, triton; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_debugging — Triton debugging guide
- uri: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton guide
- topics: debugging, compile_repair; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_cuda_semantics — PyTorch CUDA semantics
- uri: https://docs.pytorch.org/docs/2.12/notes/cuda.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, cuda; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_numerical_accuracy — PyTorch numerical accuracy note
- uri: https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, numerics; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_reproducibility — PyTorch reproducibility note
- uri: https://docs.pytorch.org/docs/2.12/notes/randomness.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, reproducibility; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_compile — PyTorch torch.compile API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.compile.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, inductor, baseline; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_allclose — PyTorch torch.allclose API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.allclose.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, tolerance; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_isclose — PyTorch torch.isclose API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.isclose.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, tolerance; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_broadcasting — PyTorch broadcasting semantics
- uri: https://docs.pytorch.org/docs/2.12/notes/broadcasting.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, broadcast; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_tensor_views — PyTorch tensor views and strides
- uri: https://docs.pytorch.org/docs/2.12/tensor_view.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, strides, layout; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_benchmark_utils — PyTorch benchmark utilities
- uri: https://docs.pytorch.org/docs/2.12/benchmark_utils.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, benchmark; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_cuda_event — PyTorch CUDA Event API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.cuda.Event.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, cuda_event; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_programming_guide — CUDA C++ programming guide
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- kind: official_doc; trust: official; license: unknown; version/date: CUDA 13.3 docs
- topics: cuda, programming_model; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_best_practices — CUDA C++ best practices guide
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- kind: official_doc; trust: official; license: unknown; version/date: CUDA 13.3 docs
- topics: cuda, benchmark, memory; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_runtime_events — CUDA Runtime API events
- uri: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- kind: official_doc; trust: official; license: unknown; version/date: CUDA docs
- topics: cuda, event_timing; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_occupancy — CUDA occupancy concepts
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy-calculator
- kind: official_doc; trust: official; license: unknown; version/date: CUDA docs
- topics: cuda, occupancy; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_coalescing — CUDA global memory coalescing
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
- kind: official_doc; trust: official; license: unknown; version/date: CUDA docs
- topics: cuda, coalescing; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_shared_memory — CUDA shared memory guidance
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory
- kind: official_doc; trust: official; license: unknown; version/date: CUDA docs
- topics: cuda, shared_memory; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## nsight_compute_guide — Nsight Compute profiling guide
- uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- kind: tool_doc; trust: official; license: unknown; version/date: Nsight Compute 13.3
- topics: ncu, profiling; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## nsight_systems_guide — Nsight Systems user guide
- uri: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- kind: tool_doc; trust: official; license: unknown; version/date: Nsight Systems docs
- topics: nsys, timeline; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## nvidia_h20_export_context — NVIDIA H20 export/backend context
- uri: https://www.reuters.com/technology/nvidia-resume-h20-gpu-sales-china-ceo-says-2025-07-15/
- kind: report; trust: project_report; license: unknown; version/date: 2025-07 reporting
- topics: h20, export_context; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## nvidia_h20_security_context — H20 security/backdoor public context
- uri: https://apnews.com/article/china-nvidia-ai-chip-backdoor-security-2025
- kind: report; trust: project_report; license: unknown; version/date: 2025-07 reporting
- topics: h20, policy, security; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_no_grad — PyTorch no_grad API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.no_grad.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, validation, no_grad; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_inference_mode — PyTorch inference_mode API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.inference_mode.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, benchmark, inference; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_contiguous — PyTorch Tensor.contiguous API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.contiguous.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, layout, contiguous; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_stride_api — PyTorch Tensor.stride API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.stride.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: PyTorch 2.12
- topics: pytorch, layout, stride; backends: nvidia_cuda, cpu; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_block_pointer_api — Triton block pointer APIs
- uri: https://triton-lang.org/main/python-api/triton.language.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, block_pointer, legality; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_compile_api — Triton compilation API context
- uri: https://triton-lang.org/main/python-api/triton.compiler.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, compiler, legality; backends: nvidia_cuda, amd_rocm; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_memory_hierarchy — CUDA memory hierarchy context
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy
- kind: official_doc; trust: official; license: unknown; version/date: CUDA docs
- topics: cuda, memory_hierarchy; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## ncu_metrics_reference — Nsight Compute metrics reference
- uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
- kind: tool_doc; trust: official; license: unknown; version/date: Nsight Compute 13.3
- topics: ncu, metrics, profile_attribution; backends: nvidia_cuda; operators: common
- candidate_fact_1: Separate PyTorch reference semantics from backend-instance optimization.
- candidate_fact_2: Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- candidate_fact_3: Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- candidate_fact_4: CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- candidate_fact_5: Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.
