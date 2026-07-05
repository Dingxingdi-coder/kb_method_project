# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A1_plain_rag`
Phase: `generate`
Task: `round1_matmul_matmul_m512_n512_k512_bf16`
Operator: `matmul` / `matmul`
Shape: `{'K': 512, 'M': 512, 'N': 512}`
DType: `bf16`
Interface: `candidate(a, b) -> c`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
