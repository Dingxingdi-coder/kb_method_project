---
archive_id: round2_pytorch_reduction
source_ids:
  - pytorch_reduction_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: bsd_3_clause
---

# PyTorch reduction contract locators

Official locators:
- `torch.sum`: https://docs.pytorch.org/docs/2.13/generated/torch.sum.html
- `torch.mean`: https://docs.pytorch.org/docs/2.13/generated/torch.mean.html
- `torch.std`: https://docs.pytorch.org/docs/2.13/generated/torch.std.html
- `torch.min`: https://docs.pytorch.org/docs/2.13/generated/torch.min.html
- `torch.argmax`: https://docs.pytorch.org/docs/2.13/generated/torch.argmax.html

Source-grounded notes:
- Reduction semantics depend on the declared dimension or dimensions, `keepdim`, optional dtype, and the resulting output shape.
- Mean divides by the number of logically reduced elements, not by a padded implementation block size.
- Standard deviation uses the task-declared correction. Current PyTorch exposes `correction`; the historical `unbiased=True` behavior corresponds to correction 1.
- `argmax` returns a LongTensor and, when multiple maxima exist, returns the index of the first maximal value.
- `min(input, dim=...)` returns both values and indices and uses the first minimal index for ties.

Applicability:
- `sum`, `mean`, `std`, `min`, `argmax`, `fused_sum_std`.

Limitations:
- Exact empty-reduction, NaN, complex-dtype, and tuple-dimension behavior must be read from the public task OpSpec.
- No H20 reduction schedule or throughput is asserted.
