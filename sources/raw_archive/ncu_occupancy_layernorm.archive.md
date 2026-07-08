---
source_id: ncu_occupancy_layernorm
archive_type: selected_source_snapshot
uri: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy
license: unknown
retrieved_at: 2026-07-06
---
# Partial archive snapshot: Nsight occupancy metrics for layernorm

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- LayerNorm forward computes mean and variance over the normalized last dimension.
- eps is inside the square-root denominator for the PyTorch-style contract.
- gamma and beta index the normalized dimension and must broadcast correctly.
- fp32 accumulation protects fp16/bf16 variance accuracy.
- Large hidden sizes can create register-pressure and occupancy tradeoffs.
