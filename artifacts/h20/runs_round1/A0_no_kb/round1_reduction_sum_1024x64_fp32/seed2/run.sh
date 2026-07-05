#!/usr/bin/env bash
set -euo pipefail
python /data/dxd/kb_method_project/experiments/h20/harness.py --task task.json --candidate candidate.py --out-dir . --hidden-tests /data/dxd/kb_method_project/artifacts/h20/tasks_round1/_hidden/round1_reduction_sum_1024x64_fp32.hidden.json
