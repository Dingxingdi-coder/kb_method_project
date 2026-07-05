# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A2_rulebook`
Phase: `generate`
Task: `round1_layernorm_layernorm_forward_4096x2048_fp16`
Operator: `layernorm` / `layernorm_forward`
Shape: `[4096, 2048]`
DType: `fp16`
Interface: `candidate(x, gamma, beta, eps) -> y`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
