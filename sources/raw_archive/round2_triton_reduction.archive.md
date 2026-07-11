---
archive_id: round2_triton_reduction
source_ids:
  - triton_reduction_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: mit
---

# Triton reduction primitive locators

Official locators:
- `tl.sum`: https://triton-lang.org/main/python-api/generated/triton.language.sum.html
- `tl.max`: https://triton-lang.org/main/python-api/generated/triton.language.max.html
- `tl.argmax`: https://triton-lang.org/main/python-api/generated/triton.language.argmax.html
- `tl.load`: https://triton-lang.org/main/python-api/generated/triton.language.load.html

Source-grounded notes:
- Masked loads need an `other` value that is neutral for the reduction: zero for sums, negative infinity or the dtype minimum for max-like reductions, and positive infinity or the dtype maximum for min-like reductions.
- `tl.argmax` exposes left tie-breaking. This can represent PyTorch's first-index tie behavior only when logical index order and reduction order are mapped consistently.
- A block reduction is an implementation primitive, not a guarantee that one program per row is optimal.
- Non-power-of-two tails must not contribute padded values to sums, counts, extrema, or index selection.

Applicability:
- Row/block reductions and softmax subreductions.

Limitations:
- Warp count, block size, multi-stage/hierarchical reduction, and atomic strategies are candidate tuning decisions requiring local H20 measurement.
