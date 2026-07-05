#!/usr/bin/env bash
set -euo pipefail
python /data/dxd/kb_method_project/experiments/h20/harness.py --task task.json --candidate candidate.py --out-dir . --hidden-tests /data/dxd/kb_method_project/artifacts/h20/tasks_round1/_hidden/round1_softmax_row_softmax_1024x257_fp16.hidden.json --warmup 100 --repeats 500
