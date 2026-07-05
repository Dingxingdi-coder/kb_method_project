# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `A3_ecc_kb`
Phase: `generate`
Task: `round1_matmul_matmul_m4096_n256_k1024_fp16`
Operator: `matmul` / `matmul`
Shape: `{'K': 1024, 'M': 4096, 'N': 256}`
DType: `fp16`
Interface: `candidate(a, b) -> c`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
