# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A3_ecc_kb`
Phase: `generate`
Task: `round1_matmul_matmul_m256_n4096_k1024_bf16`
Operator: `matmul` / `matmul`
Shape: `{'K': 1024, 'M': 256, 'N': 4096}`
DType: `bf16`
Interface: `candidate(a, b) -> c`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
