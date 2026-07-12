---
archive_id: round2_pytorch_normalization
source_ids:
  - pytorch_normalization_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: bsd_3_clause
---

# PyTorch normalization and dropout contract locators

Official locators:
- `layer_norm`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.layer_norm.html
- `RMSNorm`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html
- `dropout`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.dropout.html
- GELU/SiLU: corresponding PyTorch 2.13 functional pages

Source-grounded notes:
- LayerNorm normalizes the final `normalized_shape` dimensions using a mean and variance, then applies optional weight and bias with epsilon in the denominator.
- RMSNorm uses a root-mean-square statistic and does not subtract the mean. Reusing centered LayerNorm code changes the operator.
- Dropout randomly zeros elements with probability `p` when `training=True`; evaluation behavior must be represented explicitly rather than silently deleting dropout.
- Activation approximation and fused-subgraph order remain part of the task contract.

Applicability:
- W8A8 layernorm and the five fused normalization tasks in the expanded pilot.

Limitations:
- These pages do not define the expanded task's W8A8 quantization scale/rounding policy; that must come from public task JSON/OpSpec.
- No H20 fusion speedup is asserted.
