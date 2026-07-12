---
archive_id: round2_pytorch_matmul
source_ids:
  - pytorch_matmul_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: bsd_3_clause
---

# PyTorch matmul-family contract locators

Official locators:
- `matmul`: https://docs.pytorch.org/docs/2.13/generated/torch.matmul.html
- `addmm`: https://docs.pytorch.org/docs/2.13/generated/torch.addmm.html
- `bmm`: https://docs.pytorch.org/docs/2.13/generated/torch.bmm.html
- `tril`: https://docs.pytorch.org/docs/2.13/generated/torch.tril.html

Source-grounded notes:
- `matmul` changes output rank with 1-D inputs: vector-vector returns a scalar, matrix-vector returns a vector, and higher-rank batch dimensions broadcast.
- `addmm` computes `beta * input + alpha * (mat1 @ mat2)` and the input addend can broadcast to the 2-D output.
- `bmm` is batched matrix multiplication with a task-defined batch contract; it is not interchangeable with arbitrary broadcasting unless the task uses `matmul`.
- `tril` keeps entries on/below the selected diagonal and zeros entries above it. The diagonal offset and whether masking occurs before or after a contraction/scale are part of the fused task.

Applicability:
- `matmul`, fp16/bf16 matmul, `addmm`, `matrix_vector_dot`, `tril_mm_and_scale`.

Limitations:
- TensorFloat32, sparse, and backend-specific precision modes must follow the public task and tolerance.
- No custom-kernel performance advantage over PyTorch/cuBLAS is claimed.
