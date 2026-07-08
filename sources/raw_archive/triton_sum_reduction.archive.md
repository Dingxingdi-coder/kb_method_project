---
source_id: triton_sum_reduction
archive_type: selected_source_snapshot
uri: https://triton-lang.org/main/python-api/generated/triton.language.sum.html
license: mit
retrieved_at: 2026-07-06
---
# Partial archive snapshot: Triton tl.sum reduction API

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Row reduction must reduce the declared axis and use the neutral element for padded lanes.
- fp16/bf16 sum usually needs fp32 accumulation when the OpSpec or PyTorch reference requires it.
- Tail masks are mandatory for non-power-of-two row lengths.
- Memory bandwidth is often the primary limiter after correctness.
- One-program-per-row is a starting skeleton, not a universal H20 optimum.
