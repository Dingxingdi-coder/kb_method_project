---
archive_id: round2_pytorch_pointwise
source_ids:
  - pytorch_pointwise_contracts_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: bsd_3_clause
---

# PyTorch pointwise contract locators

Official locators:
- `torch.add`: https://docs.pytorch.org/docs/2.13/generated/torch.add.html
- `torch.mul`: https://docs.pytorch.org/docs/2.13/generated/torch.mul.html
- `torch.nn.functional.gelu`: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.gelu.html
- broadcasting note: https://docs.pytorch.org/docs/2.13/notes/broadcasting.html
- `Tensor.expand`: https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.expand.html

Source-grounded notes:
- `torch.add(input, other, alpha=..., out=...)` computes `input + alpha * other`; inputs follow broadcasting and type-promotion rules. An `out` tensor is part of the public contract when exposed by the task.
- `torch.mul` is elementwise and broadcastable; it also exposes `out`.
- Broadcast dimensions align from the trailing dimensions. An expanded singleton dimension may have zero stride and may alias one storage location across many logical output positions.
- GELU exact and tanh-approximate modes are distinct contracts. The task's `approximate` mode and output dtype/tolerance must be preserved.

Applicability:
- `add`, `mul`, `gelu_fp16`, `gelu_bf16`, `fused_add_gelu`, `fused_mul_sub`.

Limitations:
- These pages define semantics, not H20 performance.
- They do not prove that fp32 activation intermediates are always fastest; use fp32 as a correctness repair candidate when low-precision error exceeds the task tolerance.
