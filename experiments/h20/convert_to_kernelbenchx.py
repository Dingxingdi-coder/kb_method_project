#!/usr/bin/env python3
"""Export H20 candidates that have a declared KernelBench-X overlap.

The exporter is intentionally conservative. It does not rewrite candidate code or
synthesize missing KernelBench-X semantics; it only copies a candidate that already
exports the required top-level KernelBench-X entrypoint.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, sha256_file, write_json  # noqa: E402


def kbx_mapping(task: dict[str, Any]) -> tuple[str, str] | None:
    kbx = task.get("kernelbenchx")
    if isinstance(kbx, dict) and kbx.get("task_file") and kbx.get("entrypoint"):
        return Path(str(kbx["task_file"])).name, str(kbx["entrypoint"])
    op_family = task.get("op_family")
    op_name = task.get("op_name")
    dtype = str(task.get("dtype") or "")
    if op_family == "softmax" and op_name == "row_softmax":
        return "softmax.py", "softmax"
    if op_family == "reduction" and op_name == "sum":
        return "sum.py", "sum"
    if op_family == "matmul":
        if dtype in {"fp16", "float16"}:
            return "matmul_fp16.py", "matmul_fp16"
        if dtype in {"bf16", "bfloat16"}:
            return "matmul_bf16.py", "matmul_bf16"
    return None


def exports_name(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return True
    return False


def iter_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for path in paths:
        if (path / "task.json").exists() and (path / "candidate.py").exists():
            run_dirs.append(path)
            continue
        if path.is_dir():
            run_dirs.extend(p for p in path.rglob("task.json") if (p.parent / "candidate.py").exists())
    return sorted({p if p.is_dir() else p.parent for p in run_dirs})


def unique_out_dir(out_root: Path, run_dir: Path) -> Path:
    base = run_dir.name or "run"
    out = out_root / base
    if not out.exists():
        return out
    suffix = sha256_file(run_dir / "candidate.py")[:8] or "candidate"
    return out_root / f"{base}_{suffix}"


def export_one(run_dir: Path, out_root: Path, force: bool) -> dict[str, Any]:
    task_path = run_dir / "task.json"
    candidate_path = run_dir / "candidate.py"
    task = read_json(task_path)
    mapping = kbx_mapping(task)
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task_id": task.get("task_id"),
        "op_family": task.get("op_family"),
        "op_name": task.get("op_name"),
        "dtype": task.get("dtype"),
        "candidate_hash": sha256_file(candidate_path),
        "status": "skipped",
    }
    if mapping is None:
        record["reason"] = "no_declared_kernelbenchx_overlap"
        return record

    kbx_file, entrypoint = mapping
    source = candidate_path.read_text(encoding="utf-8")
    record["kernelbenchx_file"] = kbx_file
    record["kernelbenchx_entrypoint"] = entrypoint
    if not exports_name(source, entrypoint):
        record["reason"] = f"missing_top_level_entrypoint:{entrypoint}"
        return record

    out_dir = unique_out_dir(out_root, run_dir)
    if out_dir.exists():
        if not force:
            record["reason"] = f"output_exists:{out_dir}"
            return record
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / kbx_file
    out_file.write_text(source, encoding="utf-8")
    write_json(
        out_dir / "h20_kernelbenchx_export.json",
        {
            "source_run_dir": str(run_dir),
            "source_task_json": str(task_path),
            "source_candidate": str(candidate_path),
            "task_id": task.get("task_id"),
            "kernelbenchx_file": kbx_file,
            "kernelbenchx_entrypoint": entrypoint,
            "candidate_hash": sha256_file(candidate_path),
        },
    )
    record["status"] = "exported"
    record["out_dir"] = str(out_dir)
    record["out_file"] = str(out_file)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="H20 run/workspace directories, or roots containing run directories.")
    parser.add_argument("--out", required=True, help="Output root for KernelBench-X submissions.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-run export directories.")
    args = parser.parse_args()

    run_dirs = iter_run_dirs([Path(p).resolve() for p in args.runs])
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    records = [export_one(run_dir, out_root, args.force) for run_dir in run_dirs]
    write_json(out_root / "manifest.json", {"schema_version": "0.1", "exports": records})

    exported = sum(1 for r in records if r.get("status") == "exported")
    skipped = len(records) - exported
    print(f"KernelBench-X exports: exported={exported} skipped={skipped} manifest={out_root / 'manifest.json'}")
    return 0 if exported else 1


if __name__ == "__main__":
    raise SystemExit(main())
