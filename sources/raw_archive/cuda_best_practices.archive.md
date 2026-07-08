---
source_id: cuda_best_practices
archive_type: selected_source_snapshot
uri: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
license: unknown
retrieved_at: 2026-07-06
---
# Partial archive snapshot: CUDA C++ best practices guide

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- Separate PyTorch reference semantics from backend-instance optimization.
- Record H20 as a runtime-detected backend instance; do not hardcode unpublished device constants.
- Triton legality constraints such as constexpr meta-parameters, block pointers, masks, and tl.dot contracts are version scoped.
- CUDA/Nsight profiling should be summarized after correctness and interpreted with benchmark context.
- Performance claims require baseline type, repeated timing, p50/p95, and hidden correctness first.
