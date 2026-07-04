#!/usr/bin/env python3
"""Aggregate H20 MVP run results into CSV and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_result(path: Path) -> dict[str, Any]:
    doc = load_json(path) or {}
    parts = path.parts
    row: dict[str, Any] = {"result_path": str(path), "workspace": str(path.parent)}
    # Expected layout: root/group/task/seed/results.json. Fall back gracefully.
    if len(parts) >= 5:
        row["group"] = parts[-4]
        row["task_name"] = parts[-3]
        row["seed"] = parts[-2]
    task = doc.get("task", {})
    row.update({
        "op_family": task.get("op_family"),
        "op_name": task.get("op_name"),
        "dtype": task.get("dtype"),
        "shape": json.dumps(task.get("shape"), ensure_ascii=False),
        "compile_status": (doc.get("compile") or {}).get("status"),
        "smoke_status": ((doc.get("correctness") or {}).get("smoke") or {}).get("status"),
        "quick_status": ((doc.get("correctness") or {}).get("quick") or {}).get("status"),
        "hidden_status": ((doc.get("correctness") or {}).get("hidden") or {}).get("status"),
        "robustness_status": ((doc.get("correctness") or {}).get("robustness") or {}).get("status"),
    })
    bench = doc.get("benchmark") or {}
    row.update({
        "latency_p50_ms": bench.get("latency_p50_ms"),
        "latency_p95_ms": bench.get("latency_p95_ms"),
        "speedup_vs_eager_p50": bench.get("speedup_vs_eager_p50"),
        "speedup_vs_torch_compile_p50": bench.get("speedup_vs_torch_compile_p50"),
    })
    cost = doc.get("cost") or {}
    row.update({
        "iterations": cost.get("iterations"),
        "wall_time_s": cost.get("wall_time_s"),
        "agent_tokens_in": cost.get("agent_tokens_in"),
        "agent_tokens_out": cost.get("agent_tokens_out"),
        "gpu_benchmark_runs": cost.get("gpu_benchmark_runs"),
    })
    return row


def as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def median(values: list[Any]) -> float | None:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def rate(rows: list[dict[str, Any]], key: str, value: str = "pass") -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get(key) == value) / len(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted(set(str(r.get("group", "unknown")) for r in rows))
    out = []
    for group in groups:
        gr = [r for r in rows if str(r.get("group", "unknown")) == group]
        out.append({
            "group": group,
            "runs": len(gr),
            "compile_success_rate": rate(gr, "compile_status"),
            "hidden_correctness_pass_rate": rate(gr, "hidden_status"),
            "robustness_pass_rate": rate([r for r in gr if r.get("robustness_status") is not None], "robustness_status") if any(r.get("robustness_status") is not None for r in gr) else None,
            "median_latency_p50_ms": median([r.get("latency_p50_ms") for r in gr]),
            "median_latency_p95_ms": median([r.get("latency_p95_ms") for r in gr]),
            "median_speedup_vs_eager_p50": median([r.get("speedup_vs_eager_p50") for r in gr]),
            "median_speedup_vs_torch_compile_p50": median([r.get("speedup_vs_torch_compile_p50") for r in gr]),
            "median_iterations": median([r.get("iterations") for r in gr]),
            "median_wall_time_s": median([r.get("wall_time_s") for r in gr]),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = ["# H20 MVP result summary", "", "| group | runs | compile | hidden | speedup_eager_median | iter_median | wall_s_median |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(
            f"| {r['group']} | {r['runs']} | {r['compile_success_rate']:.3f} | {r['hidden_correctness_pass_rate']:.3f} | {r.get('median_speedup_vs_eager_p50') or ''} | {r.get('median_iterations') or ''} | {r.get('median_wall_time_s') or ''} |"
        )
    lines.append("")
    lines.append("Correctness is the hard gate. Do not interpret speedup values for candidates that did not pass hidden correctness.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run directories.")
    parser.add_argument("--out", required=True, help="Output Markdown report path.")
    args = parser.parse_args()

    result_paths: list[Path] = []
    for run in args.runs:
        result_paths.extend(Path(run).glob("**/results.json"))
    rows = [flatten_result(p) for p in sorted(result_paths)]
    summary = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out.with_suffix(".runs.csv"), rows)
    write_csv(out.with_suffix(".summary.csv"), summary)
    write_md(out, summary)
    print(out)


if __name__ == "__main__":
    main()
