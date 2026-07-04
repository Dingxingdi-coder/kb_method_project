#!/usr/bin/env python3
"""Generate public task.json files and private hidden-test files for the H20 MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import safe_id, short_hash, write_json  # noqa: E402


ROUND1 = [
    ("reduction", "sum", [1024, 64], "fp32"),
    ("reduction", "sum", [1024, 257], "fp16"),
    ("reduction", "max", [4096, 1024], "fp16"),
    ("reduction", "sum", [8192, 4096], "bf16"),
    ("reduction", "max", [128, 8192], "fp32"),
    ("softmax", "row_softmax", [1024, 128], "fp16"),
    ("softmax", "row_softmax", [1024, 257], "fp16"),
    ("softmax", "row_softmax", [4096, 1024], "bf16"),
    ("softmax", "row_softmax", [8192, 2048], "fp16"),
    ("softmax", "row_softmax", [512, 4096], "fp32"),
    ("layernorm", "layernorm_forward", [1024, 768], "fp16"),
    ("layernorm", "layernorm_forward", [2048, 1024], "bf16"),
    ("layernorm", "layernorm_forward", [4096, 2048], "fp16"),
    ("layernorm", "layernorm_forward", [8192, 4096], "bf16"),
    ("layernorm", "layernorm_forward", [1024, 8192], "fp16"),
    ("matmul", "matmul", {"M": 128, "N": 128, "K": 128}, "fp16"),
    ("matmul", "matmul", {"M": 512, "N": 512, "K": 512}, "bf16"),
    ("matmul", "matmul", {"M": 1024, "N": 1024, "K": 1024}, "fp16"),
    ("matmul", "matmul", {"M": 4096, "N": 256, "K": 1024}, "fp16"),
    ("matmul", "matmul", {"M": 256, "N": 4096, "K": 1024}, "bf16"),
]

ROUND2 = [
    ("softmax", "row_softmax", [2048, 513], "fp16"),
    ("softmax", "row_softmax", [16384, 1024], "bf16"),
    ("layernorm", "layernorm_forward", [4096, 3072], "fp16"),
    ("layernorm", "layernorm_forward", [2048, 6144], "bf16"),
    ("reduction", "sum", [2048, 1536], "fp16"),
    ("reduction", "max", [16384, 255], "fp32"),
    ("matmul", "matmul", {"M": 768, "N": 768, "K": 768}, "fp16"),
    ("matmul", "matmul", {"M": 2048, "N": 512, "K": 1536}, "bf16"),
]


def shape_id(shape: Any) -> str:
    if isinstance(shape, dict):
        return f"m{shape['M']}_n{shape['N']}_k{shape['K']}"
    return "x".join(str(x) for x in shape)


def hidden_variants(op_family: str, shape: Any, dtype: str) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    if op_family in {"reduction", "softmax", "layernorm"}:
        b, n = shape
        for idx, candidate_shape in enumerate([[max(1, b // 2 + 1), n + 1 if n < 8192 else n - 1], [b + 7, max(1, n - 3)]]):
            tests.append({"shape": candidate_shape, "dtype": dtype, "layout": "contiguous", "seed_offset": idx + 100})
        tests.append({"shape": shape, "dtype": dtype, "layout": "non_contiguous", "seed_offset": 200})
    elif op_family == "matmul":
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        tests.extend([
            {"shape": {"M": max(16, m // 2 + 17), "N": n, "K": k}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
            {"shape": {"M": m, "N": max(16, n // 2 + 19), "K": max(16, k // 2 + 23)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 101},
        ])
    return tests


def op_spec(op_family: str, variant: str) -> dict[str, Any]:
    if op_family == "softmax":
        return {"signature": "y = torch.softmax(x, dim=-1)", "candidate_interface": "candidate(x) -> y", "semantics": "Subtract row max before exp; mask padded tail elements.", "tolerance": "dtype-dependent"}
    if op_family == "reduction":
        return {"signature": f"y = torch.{variant}(x, dim=-1)", "candidate_interface": "candidate(x, reduce_op) -> y", "semantics": "Reduce the last dimension with fp32 accumulation for sum unless declared otherwise.", "tolerance": "dtype-dependent"}
    if op_family == "layernorm":
        return {"signature": "y = (x - mean) / sqrt(var + eps) * gamma + beta", "candidate_interface": "candidate(x, gamma, beta, eps) -> y", "semantics": "Mean and variance over last dimension; fp32 accumulation.", "tolerance": "dtype-dependent"}
    if op_family == "matmul":
        return {"signature": "C = A @ B", "candidate_interface": "candidate(a, b) -> c", "semantics": "Matrix multiplication with declared dtype and backend default/fp32 accumulation.", "tolerance": "dtype-dependent"}
    return {}


def make_task(round_name: str, op_family: str, variant: str, shape: Any, dtype: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = safe_id(f"{round_name}_{op_family}_{variant}_{shape_id(shape)}_{dtype}")
    task = {
        "schema_version": "0.1",
        "task_id": task_id,
        "round": round_name,
        "op_family": op_family,
        "op_name": variant,
        "variant": variant,
        "shape": shape,
        "dtype": dtype,
        "op_spec": op_spec(op_family, variant),
        "op_spec_hash": short_hash([op_family, variant, shape, dtype]),
        "goal": "compile_correct_and_benchmark",
        "public_tests": [{"shape": shape, "dtype": dtype, "layout": "contiguous", "seed_offset": 0}],
        "budget": {"max_iterations": 8, "max_wall_time_minutes": 60, "warmup": 100, "repeats": 500},
        "agent_allowed_files": ["candidate.py", "notes.md"],
        "agent_forbidden_files": ["reference.py", "hidden_tests.json", "experiments/h20/harness.py"],
    }
    hidden = {"schema_version": "0.1", "task_id": task_id, "hidden_tests": hidden_variants(op_family, shape, dtype), "notes": "Do not expose this file to the Coding Agent."}
    return task, hidden


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=["round1", "round2"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--operators", default=None, help="Accepted for protocol traceability; this script uses the encoded task table.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    hidden_dir = out_dir / "_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    rows = ROUND1 if args.round == "round1" else ROUND2
    manifest: list[dict[str, str]] = []

    for op_family, variant, shape, dtype in rows:
        task, hidden = make_task(args.round, op_family, variant, shape, dtype)
        task_path = out_dir / f"{task['task_id']}.json"
        hidden_path = hidden_dir / f"{task['task_id']}.hidden.json"
        write_json(task_path, task)
        write_json(hidden_path, hidden)
        manifest.append({"task_id": task["task_id"], "task": str(task_path), "hidden": str(hidden_path)})

    write_json(out_dir / "manifest.json", {"round": args.round, "tasks": manifest})
    print(f"wrote {len(manifest)} tasks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
