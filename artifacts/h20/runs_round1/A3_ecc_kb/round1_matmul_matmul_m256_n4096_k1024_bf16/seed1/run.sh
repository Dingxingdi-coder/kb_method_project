#!/usr/bin/env bash
set -euo pipefail
python /data/dxd/kb_method_project/experiments/h20/harness.py --task task.json --candidate candidate.py --out-dir . --hidden-tests /data/dxd/kb_method_project/artifacts/h20/tasks_round1/_hidden/round1_matmul_matmul_m256_n4096_k1024_bf16.hidden.json
