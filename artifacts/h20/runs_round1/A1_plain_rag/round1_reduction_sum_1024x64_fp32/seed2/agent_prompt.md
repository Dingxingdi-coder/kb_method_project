# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A1_plain_rag`
Phase: `generate`
Task: `round1_reduction_sum_1024x64_fp32`
Operator: `reduction` / `sum`
Shape: `[1024, 64]`
DType: `fp32`
Interface: `candidate(x, reduce_op) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
