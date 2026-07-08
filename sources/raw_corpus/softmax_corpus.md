---
corpus_file: softmax_corpus
derived_from: sources/registry/sources.yaml
frozen_as: raw_corpus_v0
ingested_at: 2026-07-06
---
# softmax raw corpus notes
This cleaned Markdown corpus stores metadata, compact excerpts/locators, structured notes, applicability, limitations, candidate facts, and source links. It intentionally avoids long copyrighted source dumps.

## triton_fused_softmax — Triton fused softmax tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, softmax, fusion; backends: nvidia_cuda, amd_rocm; operators: common, softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_exp_api — Triton tl.exp API
- uri: https://triton-lang.org/main/python-api/generated/triton.language.exp.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton API
- topics: triton, exp; backends: nvidia_cuda, amd_rocm; operators: common, softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_softmax — PyTorch torch.softmax API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.softmax.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_functional_softmax — PyTorch F.softmax API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.softmax.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_log_softmax — PyTorch F.log_softmax API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.log_softmax.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_softmax_tutorial — Triton fused softmax tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_softmax_source — Triton softmax tutorial source
- uri: https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## flashattention_paper — FlashAttention paper
- uri: https://arxiv.org/abs/2205.14135
- kind: paper; trust: peer_reviewed; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## online_softmax_paper — Online normalizer calculation for softmax
- uri: https://arxiv.org/abs/1805.02867
- kind: paper; trust: peer_reviewed; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_softmax_source — PyTorch softmax native source
- uri: https://github.com/pytorch/pytorch
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## xformers_softmax_context — xFormers softmax attention context
- uri: https://github.com/facebookresearch/xformers
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_exp_softmax — Triton tl.exp API for softmax
- uri: https://triton-lang.org/main/python-api/generated/triton.language.exp.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_fast_math_context — CUDA fast math context
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## ncu_softmax_memory — Nsight memory metrics for softmax
- uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_masked_softmax — Triton masked load/store for softmax
- uri: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## softmax_benchmark_note — Public softmax benchmark context
- uri: https://github.com/triton-lang/triton
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## inductor_softmax_codegen — TorchInductor softmax codegen context
- uri: https://github.com/pytorch/pytorch/tree/main/torch/_inductor
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_occupancy_softmax — CUDA occupancy context for softmax
- uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy-calculator
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_next_power_of_two — Triton next_power_of_2 helper
- uri: https://triton-lang.org/main/python-api/generated/triton.next_power_of_2.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_cross_entropy_softmax — PyTorch cross entropy softmax context
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## softmax_numerical_blog — Community softmax numerical pitfalls
- uri: https://github.com/triton-lang/triton/issues
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, cpu; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## tritonbench_softmax — TritonBench softmax tasks
- uri: https://arxiv.org/abs/2502.14752
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: softmax, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: softmax
- candidate_fact_1: Stable softmax subtracts row max before exp.
- candidate_fact_2: Padded lanes must be excluded from max, denominator, and store.
- candidate_fact_3: Approximate exp is acceptable only through dtype-specific tolerance gates.
- candidate_fact_4: Fused row softmax can reduce global-memory traffic when the row fits the working set.
- candidate_fact_5: Benchmarking must follow hidden correctness and extreme-value tests.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.
