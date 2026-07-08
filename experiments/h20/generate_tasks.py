#!/usr/bin/env python3
"""Generate public task.json files and private hidden-test files for H20 experiments."""

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

EXPANDED_PILOT = [
    ("pointwise", "bias_gelu", [4096, 1024], "bf16"),
    ("pointwise", "bias_silu", [4096, 1024], "fp16"),
    ("pointwise", "residual_relu", [8192, 512], "fp16"),
    ("pointwise", "broadcast_affine", [2048, 1536], "bf16"),
    ("pointwise", "gated_silu_multiply", [4096, 768], "fp16"),
    ("pointwise", "clamp_mul_add", [8192, 257], "fp32"),
    ("reduction", "sum", [2048, 513], "fp16"),
    ("reduction", "max", [4096, 1024], "bf16"),
    ("reduction", "sum", [8192, 257], "fp32"),
    ("reduction", "max", [1024, 4096], "fp16"),
    ("reduction", "sum", [128, 8192], "bf16"),
    ("reduction", "max", [16384, 255], "fp32"),
    ("softmax", "row_softmax", [1024, 129], "fp16"),
    ("softmax", "row_softmax", [2048, 513], "bf16"),
    ("softmax", "row_softmax", [4096, 1024], "fp16"),
    ("softmax", "row_softmax", [8192, 2048], "bf16"),
    ("softmax", "row_softmax", [512, 4096], "fp32"),
    ("softmax", "row_softmax", [16384, 255], "fp16"),
    ("layernorm", "layernorm_forward", [1024, 768], "fp16"),
    ("layernorm", "layernorm_forward", [2048, 1024], "bf16"),
    ("layernorm", "layernorm_forward", [4096, 1536], "fp16"),
    ("layernorm", "layernorm_forward", [4096, 3072], "bf16"),
    ("layernorm", "layernorm_forward", [2048, 6144], "fp16"),
    ("layernorm", "layernorm_forward", [1024, 8192], "bf16"),
    ("matmul", "matmul", {"M": 128, "N": 128, "K": 128}, "fp16"),
    ("matmul", "matmul", {"M": 512, "N": 512, "K": 512}, "bf16"),
    ("matmul", "matmul", {"M": 1024, "N": 1024, "K": 1024}, "fp16"),
    ("matmul", "matmul", {"M": 4096, "N": 256, "K": 1024}, "fp16"),
    ("matmul", "matmul", {"M": 256, "N": 4096, "K": 1024}, "bf16"),
    ("matmul", "matmul", {"M": 768, "N": 768, "K": 768}, "fp16"),
    ("layout", "transpose_copy", {"M": 2048, "N": 1024}, "bf16"),
    ("layout", "gather_rows", {"rows": 4096, "width": 256, "out_rows": 2048}, "fp16"),
    ("layout", "embedding_lookup", {"vocab": 8192, "dim": 128, "batch": 256, "seq": 16}, "bf16"),
    ("layout", "scatter_add", {"nnz": 4096, "width": 128, "out_rows": 1024}, "fp32"),
    ("layout", "slice_concat", {"B": 2048, "N": 1024, "cut": 384}, "fp16"),
    ("layout", "strided_copy", {"M": 2048, "N": 768, "row_stride": 2}, "bf16"),
]


def shape_id(shape: Any) -> str:
    if isinstance(shape, dict):
        preferred = ["M", "N", "K", "B", "rows", "width", "out_rows", "vocab", "dim", "batch", "seq", "nnz", "cut"]
        parts = [f"{key.lower()}{shape[key]}" for key in preferred if key in shape]
        parts.extend(f"{key.lower()}{value}" for key, value in sorted(shape.items()) if key not in preferred)
        return "_".join(parts)
    return "x".join(str(x) for x in shape)


def hidden_variants(op_family: str, shape: Any, dtype: str) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    if op_family in {"pointwise", "reduction", "softmax", "layernorm"}:
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
    elif op_family == "layout":
        if "M" in shape and "N" in shape:
            tests.extend([
                {"shape": {"M": max(16, int(shape["M"]) // 2 + 17), "N": int(shape["N"]) + 1}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": {"M": int(shape["M"]) + 7, "N": max(16, int(shape["N"]) - 3)}, "dtype": dtype, "layout": "non_contiguous", "seed_offset": 101},
            ])
        elif "rows" in shape:
            tests.extend([
                {"shape": {"rows": max(64, int(shape["rows"]) // 2 + 17), "width": int(shape["width"]) + 1, "out_rows": max(32, int(shape["out_rows"]) // 2 + 9)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "index_int32", "seed_offset": 101},
            ])
        elif "vocab" in shape:
            tests.extend([
                {"shape": {"vocab": max(128, int(shape["vocab"]) // 2 + 19), "dim": int(shape["dim"]) + 1, "batch": max(16, int(shape["batch"]) // 2 + 5), "seq": int(shape["seq"]) + 1}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "index_int32", "seed_offset": 101},
            ])
        elif "nnz" in shape:
            tests.extend([
                {"shape": {"nnz": max(64, int(shape["nnz"]) // 2 + 13), "width": int(shape["width"]) + 1, "out_rows": max(16, int(shape["out_rows"]) // 2 + 7)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "collision_heavy", "seed_offset": 101},
            ])
        elif "B" in shape:
            tests.extend([
                {"shape": {"B": max(16, int(shape["B"]) // 2 + 11), "N": int(shape["N"]) + 1, "cut": max(1, int(shape["cut"]) // 2)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "non_contiguous", "seed_offset": 101},
            ])
    return tests


def op_spec(op_family: str, variant: str) -> dict[str, Any]:
    if op_family == "pointwise":
        interfaces = {
            "bias_gelu": "candidate(x, bias) -> y",
            "bias_silu": "candidate(x, bias) -> y",
            "residual_relu": "candidate(x, residual) -> y",
            "broadcast_affine": "candidate(x, scale, bias) -> y",
            "gated_silu_multiply": "candidate(x1, x2) -> y",
            "clamp_mul_add": "candidate(x, scale, bias, min_value, max_value) -> y",
        }
        signatures = {
            "bias_gelu": "y = gelu(x + bias)",
            "bias_silu": "y = silu(x + bias)",
            "residual_relu": "y = relu(x + residual)",
            "broadcast_affine": "y = x * scale + bias",
            "gated_silu_multiply": "y = silu(x1) * x2",
            "clamp_mul_add": "y = clamp(x, min_value, max_value) * scale + bias",
        }
        return {"signature": signatures.get(variant, variant), "candidate_interface": interfaces.get(variant, "candidate(*args) -> y"), "semantics": "Implement the fused elementwise expression; support broadcast vectors over the last dimension and preserve output dtype.", "tolerance": "dtype-dependent", "target_banned_api": ["torch.nn.functional.gelu", "torch.nn.functional.silu", "torch.gelu", "torch.silu", "torch.clamp as the main submitted computation"]}
    if op_family == "softmax":
        return {"signature": "y = torch.softmax(x, dim=-1)", "candidate_interface": "candidate(x) -> y", "semantics": "Subtract row max before exp; mask padded tail elements.", "tolerance": "dtype-dependent"}
    if op_family == "reduction":
        return {"signature": f"y = torch.{variant}(x, dim=-1)", "candidate_interface": "candidate(x, reduce_op) -> y", "semantics": "Reduce the last dimension with fp32 accumulation for sum unless declared otherwise.", "tolerance": "dtype-dependent"}
    if op_family == "layernorm":
        return {"signature": "y = (x - mean) / sqrt(var + eps) * gamma + beta", "candidate_interface": "candidate(x, gamma, beta, eps) -> y", "semantics": "Mean and variance over last dimension; fp32 accumulation.", "tolerance": "dtype-dependent"}
    if op_family == "matmul":
        return {"signature": "C = A @ B", "candidate_interface": "candidate(a, b) -> c", "semantics": "Matrix multiplication with declared dtype and backend default/fp32 accumulation.", "tolerance": "dtype-dependent"}
    if op_family == "layout":
        interfaces = {
            "transpose_copy": "candidate(x) -> y",
            "gather_rows": "candidate(x, indices) -> y",
            "embedding_lookup": "candidate(weight, indices) -> y",
            "scatter_add": "candidate(src, indices, out_rows) -> y",
            "slice_concat": "candidate(a, b, cut) -> y",
            "strided_copy": "candidate(x) -> y",
        }
        signatures = {
            "transpose_copy": "y = x.T.contiguous()",
            "gather_rows": "y = x[indices]",
            "embedding_lookup": "y = weight[indices]",
            "scatter_add": "y = zeros(out_rows, width).scatter_add(dim=0, index=indices, src=src)",
            "slice_concat": "y = torch.cat([a[:, :cut], b[:, cut:]], dim=-1)",
            "strided_copy": "y = x.contiguous() for a strided input view",
        }
        return {"signature": signatures.get(variant, variant), "candidate_interface": interfaces.get(variant, "candidate(*args) -> y"), "semantics": "Implement the declared layout or irregular-memory transformation without reading hidden cases; handle bounds, strides, index dtype, and collisions where applicable.", "tolerance": "dtype-dependent", "target_banned_api": ["torch.transpose/permute/contiguous as the submitted target copy", "torch.gather/index_select/embedding as the submitted target gather", "torch.scatter_add as the submitted target scatter", "torch.cat as the submitted target concat"]}
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
    parser.add_argument("--round", choices=["round1", "round2", "expanded_pilot"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--operators", default=None, help="Accepted for protocol traceability; this script uses the encoded task table.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    hidden_dir = out_dir / "_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    if args.round == "round1":
        rows = ROUND1
    elif args.round == "round2":
        rows = ROUND2
    else:
        rows = EXPANDED_PILOT
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
