# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A2_rulebook`
Phase: `generate`
Task: `round1_softmax_row_softmax_1024x257_fp16`
Operator: `softmax` / `row_softmax`
Shape: `[1024, 257]`
DType: `fp16`
Interface: `candidate(x) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
