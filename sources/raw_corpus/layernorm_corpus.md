---
corpus_file: layernorm_corpus
derived_from: sources/registry/sources.yaml
frozen_as: raw_corpus_v0
ingested_at: 2026-07-06
---
# layernorm raw corpus notes
This cleaned Markdown corpus stores metadata, compact excerpts/locators, structured notes, applicability, limitations, candidate facts, and source links. It intentionally avoids long copyrighted source dumps.

## triton_layernorm_tutorial — Triton layernorm tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- kind: official_doc; trust: official; license: unknown; version/date: Triton main docs
- topics: triton, layernorm, reductions; backends: nvidia_cuda, amd_rocm; operators: common, layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_layernorm_module — PyTorch LayerNorm module
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_layer_norm_functional — PyTorch F.layer_norm API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.layer_norm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_native_layer_norm — PyTorch native_layer_norm API
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.native_layer_norm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_layernorm_tutorial_detail — Triton layernorm tutorial
- uri: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_layernorm_source — Triton layernorm tutorial source
- uri: https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## apex_layernorm — NVIDIA Apex fused layernorm
- uri: https://github.com/NVIDIA/apex/tree/master/apex/normalization
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_welford — Welford variance algorithm
- uri: https://dl.acm.org/doi/10.1145/359146.359153
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_layernorm_source — PyTorch LayerNorm native source
- uri: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## oneflow_layernorm — OneFlow LayerNorm CUDA source
- uri: https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/layer_norm_kernel.cu
- kind: community; trust: community; license: apache_2; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_register_pressure — CUDA register pressure guidance
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## ncu_occupancy_layernorm — Nsight occupancy metrics for layernorm
- uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## triton_masked_load_layernorm — Triton masked load pattern for layernorm
- uri: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_eps_layernorm — PyTorch LayerNorm eps semantics
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## pytorch_affine_layernorm — PyTorch LayerNorm affine semantics
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html
- kind: official_doc; trust: official; license: bsd_3_clause; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## cuda_bandwidth_layernorm — CUDA effective bandwidth for layernorm
- uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effective-bandwidth-calculation
- kind: official_doc; trust: official; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## transformers_layernorm — Transformers LayerNorm usage context
- uri: https://github.com/huggingface/transformers
- kind: community; trust: community; license: apache_2; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## tritonbench_layernorm — TritonBench layernorm tasks
- uri: https://arxiv.org/abs/2502.14752
- kind: official_doc; trust: official; license: mit; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, amd_rocm; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## inductor_layernorm_codegen — TorchInductor layernorm codegen context
- uri: https://github.com/pytorch/pytorch/tree/main/torch/_inductor
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## layernorm_forum_failures — Community layernorm failure modes
- uri: https://github.com/triton-lang/triton/issues
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.

## torch_compile_layernorm — torch.compile layernorm baseline
- uri: https://docs.pytorch.org/docs/2.12/generated/torch.compile.html
- kind: community; trust: community; license: unknown; version/date: current or publication date as noted by source
- topics: layernorm, h20_mvp; backends: nvidia_cuda, cpu; operators: layernorm
- candidate_fact_1: LayerNorm forward computes mean and variance over the normalized last dimension.
- candidate_fact_2: eps is inside the square-root denominator for the PyTorch-style contract.
- candidate_fact_3: gamma and beta index the normalized dimension and must broadcast correctly.
- candidate_fact_4: fp32 accumulation protects fp16/bf16 variance accuracy.
- candidate_fact_5: Large hidden sizes can create register-pressure and occupancy tradeoffs.
- applicability: only within declared OpSpec dtype, shape, layout, phase, and backend/toolchain scope; H20 performance transfer requires repository measurement.
- limitations: this entry is a cleaned note and locator, not an unbounded copy of the upstream document.
