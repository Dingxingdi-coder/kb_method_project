# Agent notes

- Implemented row softmax with a Triton kernel using dynamic `x.shape[-1]`.
- Kernel computes row max and sum inside Triton with masked padded tail elements.
- Verified with `python -m py_compile`; GPU harness was not run.
