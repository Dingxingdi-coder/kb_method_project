# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A0_no_kb`
Phase: `generate`
Task: `round1_reduction_sum_1024x257_fp16`
Operator: `reduction` / `sum`
Shape: `[1024, 257]`
DType: `fp16`
Interface: `candidate(x, reduce_op) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
