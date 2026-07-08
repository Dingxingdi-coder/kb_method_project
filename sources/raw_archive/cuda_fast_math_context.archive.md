---
source_id: cuda_fast_math_context
archive_type: selected_source_snapshot
uri: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
license: unknown
retrieved_at: 2026-07-06
---
# Partial archive snapshot: CUDA fast math context

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Stable softmax subtracts row max before exp.
- Padded lanes must be excluded from max, denominator, and store.
- Approximate exp is acceptable only through dtype-specific tolerance gates.
- Fused row softmax can reduce global-memory traffic when the row fits the working set.
- Benchmarking must follow hidden correctness and extreme-value tests.
