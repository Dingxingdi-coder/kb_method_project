# Agent notes

- Implemented row softmax over the last dimension without calling torch.softmax.
- CUDA path uses a Triton per-row kernel with explicit tail masks and fp32 reduction math; CPU/non-Triton fallback uses torch primitives with fp32 accumulation.
