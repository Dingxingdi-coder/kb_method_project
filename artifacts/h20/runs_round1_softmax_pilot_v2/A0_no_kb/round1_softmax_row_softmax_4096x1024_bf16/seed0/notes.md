# Agent notes

- Implemented `candidate(x)` with a Triton row-wise softmax kernel.
- The wrapper makes a contiguous working copy for non-contiguous inputs, supports arbitrary leading dimensions, and masks non-power-of-two row widths.
- Validation: `python -m py_compile candidate.py`.
