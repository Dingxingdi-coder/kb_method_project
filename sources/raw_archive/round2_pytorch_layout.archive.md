---
archive_id: round2_pytorch_layout
source_ids:
  - pytorch_layout_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: bsd_3_clause
---

# PyTorch layout/indexing contract locators

Official locators:
- tensor views/strides: https://docs.pytorch.org/docs/2.13/tensor_view.html
- `index_select`: https://docs.pytorch.org/docs/2.13/generated/torch.index_select.html
- `permute_copy`: https://docs.pytorch.org/docs/2.13/generated/torch.permute_copy.html
- `scatter`: https://docs.pytorch.org/docs/2.13/generated/torch.scatter.html
- `masked_select`: https://docs.pytorch.org/docs/2.13/generated/torch.masked_select.html
- `Tensor.expand`: https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.expand.html
- `where`: https://docs.pytorch.org/docs/2.13/generated/torch.where.html

Source-grounded notes:
- A view may be non-contiguous; storage offsets must be reconstructed from logical coordinates and strides.
- `index_select` selects along one dimension using a 1-D integer index tensor and returns a new tensor with the selected dimension replaced by the index length.
- `permute_copy` is a materialized copy, not a metadata-only view.
- Scatter overwrite and scatter-add are different contracts. Duplicate destination indices require the task's documented policy and can be nondeterministic.
- `masked_select` returns a one-dimensional compacted result, and its mask may broadcast against input.
- `expand` can expose zero-stride dimensions; `where` requires its three operands to be broadcastable.

Applicability:
- All six expanded layout/indexing tasks.

Limitations:
- Index sorting/uniqueness, collision rate, output true-count, and coalescing are workload-dependent and cannot be assumed.
- No H20 compaction, atomic, or tiled-copy speedup is claimed.
