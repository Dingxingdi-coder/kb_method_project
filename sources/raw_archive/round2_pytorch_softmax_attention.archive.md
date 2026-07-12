---
archive_id: round2_pytorch_softmax_attention
source_ids:
  - pytorch_softmax_attention_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: bsd_3_clause
---

# PyTorch softmax, cross-entropy, and attention contract locators

Official locators:
- `softmax`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.softmax.html
- `log_softmax`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.log_softmax.html
- `cross_entropy`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html
- `scaled_dot_product_attention`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html

Source-grounded notes:
- Softmax/log-softmax apply along the declared dimension. A requested dtype casts the input before the operation and can prevent overflow.
- Computing `log(softmax(x))` as two independent operations is both less stable and generally less efficient than the dedicated log-softmax formulation.
- Stable softmax uses a row maximum before exponentiation. Log-softmax can reuse the maximum and log-sum-exp rather than materialize probabilities.
- Cross entropy includes target representation, optional class weights, `ignore_index`, reduction mode, and label smoothing. These are semantic conditions, not tuning knobs.
- Scaled dot-product attention applies a scale (defaulting to inverse square root of key dimension when omitted), optional boolean/additive attention masking, causal masking, softmax, dropout, and value aggregation in a defined order. Dropout probability must be set to zero by the caller for evaluation behavior.

Applicability:
- `softmax`, fused softmax/log-softmax tasks, cross-entropy fusions, and `attention`.

Limitations:
- The official operator may dispatch to optimized backend kernels. This is not permission to use a high-level fallback in the repository's kernel-authoring candidate.
- No H20 FlashAttention or CE-fusion performance claim is imported.
