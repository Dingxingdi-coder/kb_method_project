---
source_id: triton_fused_softmax
archive_type: selected_source_snapshot
uri: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
license: unknown
retrieved_at: 2026-07-06
---
# Partial archive snapshot: Triton fused softmax tutorial

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Stable softmax subtracts row max before exp.
- Padded lanes must be excluded from max, denominator, and store.
- Approximate exp is acceptable only through dtype-specific tolerance gates.
- Fused row softmax can reduce global-memory traffic when the row fits the working set.
- Benchmarking must follow hidden correctness and extreme-value tests.
