---
source_id: triton_layernorm_source
archive_type: selected_source_snapshot
uri: https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py
license: mit
retrieved_at: 2026-07-06
---
# Partial archive snapshot: Triton layernorm tutorial source

This is a limited audit snapshot, not a full upstream copy.

## Preserved facts
- LayerNorm forward computes mean and variance over the normalized last dimension.
- eps is inside the square-root denominator for the PyTorch-style contract.
- gamma and beta index the normalized dimension and must broadcast correctly.
- fp32 accumulation protects fp16/bf16 variance accuracy.
- Large hidden sizes can create register-pressure and occupancy tradeoffs.
