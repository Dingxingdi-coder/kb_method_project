---
source_id: triton_softmax_source
archive_type: selected_source_snapshot
uri: https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py
license: mit
retrieved_at: 2026-07-06
---
# Partial archive snapshot: Triton softmax tutorial source

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Stable softmax subtracts row max before exp.
- Padded lanes must be excluded from max, denominator, and store.
- Approximate exp is acceptable only through dtype-specific tolerance gates.
- Fused row softmax can reduce global-memory traffic when the row fits the working set.
- Benchmarking must follow hidden correctness and extreme-value tests.
