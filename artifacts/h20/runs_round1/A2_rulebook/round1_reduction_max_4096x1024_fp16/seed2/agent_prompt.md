# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A2_rulebook`
Phase: `generate`
Task: `round1_reduction_max_4096x1024_fp16`
Operator: `reduction` / `max`
Shape: `[4096, 1024]`
DType: `fp16`
Interface: `candidate(x, reduce_op) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
