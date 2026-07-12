---
corpus_file: expanded_pointwise_round2_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_round2_sources.yaml
frozen_as: raw_corpus_v0_expanded_pilot_round2
ingested_at: 2026-07-11
updated_at: 2026-07-11
---

# Expanded pilot round2: pointwise / fused memory-bound

This file is source-visible input for A1 and the sole evidence base, together with its registry/archive records, for the round2 pointwise claims and KB entries. It is not an experiment result.

## Actual task family

Source `kbx_round2_task_manifest` lists:
`add`, `mul`, `gelu_fp16`, `gelu_bf16`, `fused_add_gelu`, `fused_mul_sub`.

No hidden shape, hidden test, or benchmark value is recorded here.

## Semantic and correctness rules

Sources: `pytorch_pointwise_contracts_round2`, `triton_pointwise_tutorial_round2`, `h20_round2_authoring_protocol`.

- `add` must implement `input + alpha * other` when the public signature exposes `alpha`. Preserve broadcasting, type promotion, output dtype, and `out` behavior declared by the task.
- `mul` is elementwise and broadcastable. Do not assume both operands are same-shape contiguous tensors.
- Broadcasting aligns dimensions from the right. For singleton/expanded dimensions, address the same storage element across the expanded logical dimension; a zero stride is a valid representation.
- `gelu_fp16`, `gelu_bf16`, and `fused_add_gelu` must preserve the declared GELU approximation mode. Exact and tanh-approximate GELU are not interchangeable.
- A conservative correctness repair for activation mismatch is to compute nonlinear intermediates in fp32 and cast once at the output. This is not a claim that fp32 intermediates are always fastest.
- `fused_mul_sub` must preserve the public arithmetic order and dtype behavior; do not silently reassociate operations across a tolerance-sensitive boundary.
- A flat one-dimensional Triton skeleton uses `program_id`, `arange`, and a tail predicate. Apply the predicate to every potentially out-of-bounds load and store.
- In kernel-authoring mode, the submitted target path must not call `torch.add`, `torch.mul`, `torch.nn.functional.gelu`, `torch.nn.functional.silu`, or another high-level target primitive to implement the operation.

## Candidate performance guidance

Sources: `triton_pointwise_tutorial_round2`, `cuda_memory_access_round2`, `h20_round2_measurement_protocol`.

- Map adjacent lanes to adjacent output elements for contiguous output and contiguous main operands. Reconstruct broadcast/strided operand addresses separately.
- Fuse `add + GELU` or `mul - sub` only when fusion removes an intermediate allocation/write/read or a separate launch. The semantic contract must still be implemented in one legal low-level path.
- Avoid materializing expanded/broadcast operands. Compute zero-stride addresses in the kernel.
- Consider vectorized or wider contiguous transactions only after alignment, tail, dtype, and stride cases are correct. Concrete vector width is a tuning knob, not a stable rule.
- Measure the exact eager baseline and the legal custom candidate with the same warmup/repeat settings. Documentation does not establish that a custom kernel wins on H20.

## Continue/stop boundary

- Continue when profiling exposes an actionable defect: non-coalesced main loads/stores, unnecessary temporary materialization, multiple launches for a fusible chain, excessive register pressure, or a launch/configuration mismatch with repeatable impact.
- For standalone `add` or `mul`, stop broad tuning when the candidate is legal and hidden-correct, repeated p50/p95 measurements fail to beat the exact eager baseline beyond noise, and there is no fusion opportunity or profile-attributed defect.
- For `fused_add_gelu` and `fused_mul_sub`, continue only while fusion actually removes traffic/launches and repeated measurements show headroom. Stop when a single-pass coalesced candidate is at a stable memory/launch-bound plateau.
- A stop decision is a local budget decision, not a proof of global optimality.

## Failure boundaries and validation

Validate:
- odd and non-power-of-two element counts;
- scalar, same-shape, and singleton-dimension broadcasting;
- non-contiguous operands when the public task permits them;
- `alpha` values other than one and `out` behavior when exposed;
- fp16/bf16 output tolerance, large positive/negative GELU inputs, exact/tanh approximation;
- aliasing restrictions for expanded outputs;
- anti-cheating/static legality before benchmark.

Common failures:
- using the output flat offset directly for a broadcast operand;
- missing a load or store tail mask;
- implementing `input + other` while ignoring `alpha`;
- returning a new allocation when the task requires `out`;
- using the wrong GELU approximation;
- calling a high-level PyTorch activation from the submitted path;
- claiming an H20 speedup from a generic Triton/CUDA document.
