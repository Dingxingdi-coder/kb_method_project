# Agent notes

- Implemented row softmax with a Triton kernel, one program per flattened row.
- Uses masked loads/stores for non-power-of-two row widths and fp32 reductions for max/sum.
- Makes a contiguous working copy when needed and preserves the input shape in the returned tensor.
