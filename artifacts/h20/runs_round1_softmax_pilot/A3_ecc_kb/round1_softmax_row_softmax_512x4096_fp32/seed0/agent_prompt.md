# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A3_ecc_kb`
Phase: `generate`
Task: `round1_softmax_row_softmax_512x4096_fp32`
Operator: `softmax` / `row_softmax`
Shape: `[512, 4096]`
DType: `fp32`
Interface: `candidate(x) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
