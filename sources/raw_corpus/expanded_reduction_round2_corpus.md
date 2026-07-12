---
corpus_file: expanded_reduction_round2_corpus
derived_from:
  - sources/registry/h20_expanded_pilot_round2_sources.yaml
frozen_as: raw_corpus_v0_expanded_pilot_round2
ingested_at: 2026-07-11
updated_at: 2026-07-11
---

# Expanded pilot round2: reduction

This file is source-visible input for A1 and the evidence base for round2 reduction claims/KB. It contains no H20 timing result.

## Actual task family

Source `kbx_round2_task_manifest` lists:
`sum`, `mean`, `std`, `min`, `argmax`, `fused_sum_std`.

## Semantic and correctness rules

Sources: `pytorch_reduction_contracts_round2`, `triton_reduction_contracts_round2`, `h20_round2_authoring_protocol`.

- Read `dim`, scalar-vs-tuple dimensions, `dim=None`, `keepdim`, optional dtype, output arity, and empty/degenerate behavior from the public task.
- `sum` and `mean` reduce the declared logical elements. `mean` divides by the true count, never a padded block size.
- Accumulate fp16/bf16 sums, means, second moments, and softmax-like statistics in fp32 when required to meet the reference tolerance; cast at the declared output boundary.
- `std` uses the declared correction. The denominator is `N - correction` subject to the reference definition; correction 0 and correction 1 are distinct.
- A two-pass mean/deviation implementation is often a simple correctness baseline. A one-pass method is acceptable only if it reproduces the task's tolerance and correction semantics.
- `argmax` returns an integer index (PyTorch LongTensor semantics unless the task overrides it) and returns the first maximal index on ties.
- `min(input, dim=...)` can return both values and indices and uses the first minimal index on ties. Do not confuse `torch.min` value+index output with a value-only reduction.
- `fused_sum_std` must reproduce both outputs independently. Sharing reads does not permit sharing the wrong denominator or returning one statistic in the wrong dtype/shape.
- Tail lanes need neutral values: zero for sums; positive infinity or the dtype maximum for minimum; negative infinity or the dtype minimum for maximum; and a validity predicate for index tie-breaking.
- `tl.argmax(tie_break_left=True)` can represent first-index behavior only when the logical reduction order is mapped consistently to lane order.

## Candidate implementation and performance guidance

Sources: `triton_reduction_contracts_round2`, `cuda_memory_access_round2`, `h20_round2_measurement_protocol`.

- One program per output row with a block reduction is a starting skeleton for a contiguous last-axis reduction, not a universal schedule.
- For long reductions or few output rows, candidate strategies include multi-program partial reductions, warp/block hierarchy, or a staged final reduction.
- For arbitrary axes, map output coordinates to input strides explicitly; a row-only skeleton is invalid when the reduced axis is not contiguous/last.
- For `fused_sum_std`, sharing a single input read can reduce traffic, but the additional statistics/register state may lower occupancy. Keep this candidate-only until H20 benchmark/profile evidence exists.
- Tune block size and warps only after masks, count, correction, dtype, shape, and tie semantics pass hidden correctness.

## Continue/stop boundary

- Continue while profiling identifies serial reduction depth, under-filled programs, uncoalesced axis traversal, excessive staged-launch cost, or register/occupancy pressure with a specific repair.
- Stop broad tuning after hidden correctness, repeated measurements, and profile attribution show a plateau across a bounded set of block/warp/staging choices.
- Do not stop if only the public contiguous-last-axis case passes; arbitrary-dim correctness remains unresolved.
- Do not promote a general CUDA/Triton schedule to a stable H20 conclusion without task-run evidence.

## Failure boundaries and validation

Validation matrix:
- `dim=None`, positive and negative dimensions, tuple dimensions if exposed;
- `keepdim=True/False`;
- length 0/1 where legal, odd lengths, non-power-of-two lengths;
- dtype override and fp16/bf16 tolerance;
- `std` correction 0, 1, and another allowed value;
- all-equal inputs and repeated extrema to test first-index ties;
- min value+index arity and argmax index dtype;
- fused sum/std checked output-by-output.

Common failures:
- dividing by padded block size;
- ignoring `correction`;
- using a neutral value that wins a min/max tail;
- returning values for argmax or dropping indices from min;
- choosing the last tied index instead of the first;
- hardcoding last-axis contiguous addressing;
- using atomic accumulation without validating nondeterministic tolerance;
- treating a generic block reduction as H20-verified.
