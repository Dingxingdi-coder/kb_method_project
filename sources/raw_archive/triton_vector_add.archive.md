---
source_id: triton_vector_add
archive_type: selected_source_snapshot
uri: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
license: unknown
retrieved_at: 2026-07-06
---
# Partial archive snapshot: Triton vector add tutorial

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Separate PyTorch reference semantics from backend-instance optimization.
- Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
