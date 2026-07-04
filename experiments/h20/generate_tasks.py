#!/usr/bin/env python3
"""Generate JSON OpSpec tasks for the H20 MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROUND1 = {
    "reduction_sum": {"op_family": "reduction", "op_name": "row_sum", "shapes": [[1024, 64], [1024, 257], [4096, 1024], [8192, 4096], [128, 8192]], "dtypes": ["fp32", "fp16", "bf16"], "axis": -1, "accumulation": "fp32"},
    "reduction_max": {"op_family": "reduction", "op_name": "row_max", "shapes": [[1024, 64], [1024, 257], [4096, 1024], [8192, 4096], [128, 8192]], "dtypes": ["fp32", "fp16", "bf16"], "axis": -1},
    "softmax": {"op_family": "softmax", "op_name": "row_softmax", "shapes": [[1024, 128], [1024, 257], [4096, 1024], [8192, 2048], [512, 4096]], "dtypes": ["fp16", "bf16", "fp32"], "axis": -1},
    "layernorm": {"op_family": "layernorm", "op_name": "layernorm_forward", "shapes": [[1024, 768], [2048, 1024], [4096, 2048], [8192, 4096], [1024, 8192]], "dtypes": ["fp16", "bf16"], "axis": -1, "eps": 1e-5, "accumulation": "fp32"},
    "matmul": {"op_family": "matmul", "op_name": "matmul", "shapes": [[128, 128, 128], [512, 512, 512], [1024, 1024, 1024], [4096, 256, 1024], [256, 4096, 1024]], "dtypes": ["fp16", "bf16"], "accumulation": "fp32_or_backend_default"},
}

ROUND2 = {
    "softmax": {"op_family": "softmax", "op_name": "row_softmax", "shapes": [[2048, 513], [16384, 1024]], "dtypes": ["fp16", "bf16", "fp32"], "axis": -1},
    "layernorm": {"op_family": "layernorm", "op_name": "layernorm_forward", "shapes": [[4096, 3072], [2048, 6144]], "dtypes": ["fp16", "bf16"], "axis": -1, "eps": 1e-5, "accumulation": "fp32"},
    "reduction_sum": {"op_family": "reduction", "op_name": "row_sum", "shapes": [[2048, 1536], [16384, 255]], "dtypes": ["fp32", "fp16", "bf16"], "axis": -1, "accumulation": "fp32"},
    "reduction_max": {"op_family": "reduction", "op_name": "row_max", "shapes": [[2048, 1536], [16384, 255]], "dtypes": ["fp32", "fp16", "bf16"], "axis": -1},
    "matmul": {"op_family": "matmul", "op_name": "matmul", "shapes": [[768, 768, 768], [2048, 512, 1536]], "dtypes": ["fp16", "bf16"], "accumulation": "fp32_or_backend_default"},
}


def task_hash(task: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def shape_name(shape: list[int], op: str) -> str:
    if op == "matmul":
        return f"m{shape[0]}_n{shape[1]}_k{shape[2]}"
    return "x".join(str(x) for x in shape)


def expand(specs: dict[str, dict[str, Any]], round_name: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for key, spec in specs.items():
        for shape in spec["shapes"]:
            for dtype in spec["dtypes"]:
                task = {k: v for k, v in spec.items() if k not in {"shapes", "dtypes"}}
                task.update({"shape": shape, "dtype": dtype, "round": round_name, "goal": "correct_then_fast", "layout": "contiguous", "backend": "triton_cuda"})
                task["name"] = f"{task['op_name']}_{shape_name(shape, task['op_family'])}_{dtype}"
                task["op_spec_hash"] = task_hash(task)
                tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=["round1", "round2"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for pilot runs.")
    args = parser.parse_args()

    specs = ROUND1 if args.round == "round1" else ROUND2
    tasks = expand(specs, args.round)
    if args.limit:
        tasks = tasks[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task in tasks:
        path = out / f"{task['name']}.json"
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append(str(path))
    (out / "manifest.json").write_text(json.dumps({"round": args.round, "tasks": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(tasks)} task(s) under {out}")


if __name__ == "__main__":
    main()
