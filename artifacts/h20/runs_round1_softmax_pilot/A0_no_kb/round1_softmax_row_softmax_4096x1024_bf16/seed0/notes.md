# Agent notes

- Implemented a Triton row-wise softmax over the last dimension using fp32 accumulation, row-max subtraction, and tail masking for non-power-of-two widths.
- Added a CPU/oversized-row fallback using explicit max/exp/sum arithmetic without calling torch.softmax.
