# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A2_rulebook`
Phase: `generate`
Task: `round1_reduction_sum_8192x4096_bf16`
Operator: `reduction` / `sum`
Shape: `[8192, 4096]`
DType: `bf16`
Interface: `candidate(x, reduce_op) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
