# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A0_no_kb`
Phase: `generate`
Task: `round1_softmax_row_softmax_4096x1024_bf16`
Operator: `softmax` / `row_softmax`
Shape: `[4096, 1024]`
DType: `bf16`
Interface: `candidate(x) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
