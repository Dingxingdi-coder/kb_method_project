# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A3_ecc_kb`
Phase: `generate`
Task: `round1_reduction_max_128x8192_fp32`
Operator: `reduction` / `max`
Shape: `[128, 8192]`
DType: `fp32`
Interface: `candidate(x, reduce_op) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
