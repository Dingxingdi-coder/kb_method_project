# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A3_ecc_kb`
Phase: `generate`
Task: `round1_layernorm_layernorm_forward_2048x1024_bf16`
Operator: `layernorm` / `layernorm_forward`
Shape: `[2048, 1024]`
DType: `bf16`
Interface: `candidate(x, gamma, beta, eps) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
