---
corpus_file: expanded_softmax_attention_round2_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_round2_sources.yaml
frozen_as: raw_corpus_v0_expanded_pilot_round2
ingested_at: 2026-07-11
updated_at: 2026-07-11
---

# Expanded pilot round2: softmax / log-softmax / cross-entropy / attention

This file is source-visible A1 evidence. It records semantics, validation, and candidate optimization boundaries, not an H20 benchmark result.

## Actual task family

Source `kbx_round2_task_manifest` lists:
`softmax`, `fused_softmax_log`, `fused_log_softmax_linear`,
`fused_repeat_interleave_log_softmax`, `fused_cross_entropy_log_softmax`, `attention`.

## Stable numerical core

Sources: `pytorch_softmax_attention_contracts_round2`, `triton_softmax_attention_tutorials_round2`.

For one reduction slice:
1. compute the maximum over valid elements;
2. subtract the maximum before exponentiation;
3. accumulate exponentials in sufficient precision;
4. divide by the valid sum for softmax, or compute `x - max - log(sum(exp(x-max)))` for log-softmax;
5. cast to the declared output dtype.

Rules:
- Reduce along the declared `dim`; do not hardcode the last dimension unless the public task guarantees it.
- When a dtype argument is exposed, it applies before softmax/log-softmax and can be required to prevent overflow.
- Tail and semantic masks must be applied before max and denominator accumulation. Masked lanes must not contribute to the maximum or sum.
- `log(softmax(x))` materialized as two operations is not the dedicated log-softmax contract; use the stable log-sum-exp form.
- `fused_softmax_log` must follow its exact public expression. If it requests a logarithm of softmax, validate special values and output domain rather than assuming it is an alias of log-softmax.

## Cross-entropy boundary

Source: `pytorch_softmax_attention_contracts_round2`.

Before fusion, preserve:
- target format (class indices or probabilities);
- class dimension and target/output shapes;
- optional class weight;
- `ignore_index`;
- reduction (`none`, `mean`, or `sum`);
- label smoothing;
- output dtype and scalar/tensor shape.

A fused cross-entropy/log-softmax kernel may avoid materializing the full log-probability tensor only when it can compute the exact target/reduction semantics. Ignored targets must not contribute to numerator or mean denominator.

## Attention boundary

Sources: `pytorch_softmax_attention_contracts_round2`, `triton_softmax_attention_tutorials_round2`.

The contract includes:
- score contraction `Q @ K^T`;
- explicit or default scale;
- optional boolean/additive attention mask;
- causal lower-triangular masking when declared;
- stable softmax;
- dropout behavior;
- value contraction with `V`;
- output shape/layout/dtype.

Apply masks before softmax. Do not combine a disallowed explicit attention mask with causal mode. Evaluation behavior must pass `dropout_p=0` according to the task; silently dropping nonzero training dropout is incorrect.

## Candidate performance guidance

- Fusing max/exp/sum/normalize avoids intermediate tensors and extra launches for row softmax; this is a general candidate supported by Triton tutorials.
- Fuse pre/post pointwise operations only when the row state and output shape fit the same program without excessive register pressure.
- For cross entropy, target-only output can avoid storing full log probabilities, but only after class-weight, ignore-index, label-smoothing, and reduction semantics are correct.
- Attention tiling can reuse Q/K/V and avoid materializing the score matrix, but causal mode, head dimension, sequence shape, dtype, and dropout create distinct schedule regimes.
- fp32 accumulation is a correctness-oriented default for low-precision logits/statistics; lower precision is a measured candidate, not a stable rule.
- All launch, tile, warp, and persistent choices remain Triton/CUDA general candidates until H20 runs validate them.

## Continue/stop boundary

- Continue while overflow/underflow, mask-order bugs, incorrect reduction shape, unnecessary score/log-probability materialization, or profile-attributed register/occupancy/launch issues remain.
- Stop broad tuning when hidden correctness is stable across numerical and mask cases, repeated benchmarks plateau, and profile evidence does not identify a targeted repair.
- For a complex fused task, stop expanding fusion when added fusion increases spills/occupancy loss or creates correctness risk without repeatable end-to-end gain.

## Validation and common failures

Validate:
- large positive/negative logits and all-equal rows;
- different positive/negative dims;
- odd row lengths and masks;
- dtype override;
- CE `ignore_index`, weights, label smoothing, reduction modes, and target forms exposed by the task;
- causal/non-causal attention, explicit boolean/additive masks, scale default/override, dropout training/eval policy;
- fused output shape after repeat-interleave or linear subexpressions.

Common failures:
- exponentiating raw logits without max subtraction;
- masked lanes entering max or denominator;
- returning softmax where log-softmax is required;
- averaging CE over ignored targets;
- applying causal mask after softmax;
- omitting scale or applying it in the wrong location;
- silently removing dropout;
- treating the attention task as only a row-softmax kernel;
- citing tutorial performance as H20 evidence.
