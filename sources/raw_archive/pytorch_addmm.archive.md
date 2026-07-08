---
source_id: pytorch_addmm
archive_type: selected_source_snapshot
uri: https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html
license: bsd_3_clause
retrieved_at: 2026-07-06
---
# Partial archive snapshot: PyTorch torch.addmm API

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Matmul must respect PyTorch M/N/K, batch, broadcasting, and optional bias boundaries.
- Triton matmul uses M/N/K tiles and tl.dot-like inner products.
- Boundary masks are required for non-multiple tile sizes.
- TF32/fp16/bf16 accumulation policy must be explicit and compared consistently.
- Tile sizes, num_warps, stages, split-K, and persistent scheduling are backend-instance knobs requiring H20 measurement.
