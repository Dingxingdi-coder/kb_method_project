#!/usr/bin/env python3
"""Aggregate H20 MVP results into CSV, JSON, and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, read_jsonl, write_json  # noqa: E402


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def bool_pass(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("status")
    return 1 if str(value).lower() in ("pass", "passed", "true", "1") else 0


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    results_path = run_dir / "results.json"
    task_path = run_dir / "task.json"
    context_path = run_dir / "context_packet.json"
    trace_path = run_dir / "trace.jsonl"
    if not results_path.exists() or not task_path.exists():
        return None
    results = read_json(results_path)
    task = read_json(task_path)
    context = read_json(context_path, default={}) if context_path.exists() else {}
    trace = read_jsonl(trace_path) if trace_path.exists() else []
    group = context.get("group") or next((p for p in run_dir.parts if p.startswith("A")), "unknown")

    compile_pass = bool_pass(get_nested(results, "compile.status"))
    hidden_pass = bool_pass(get_nested(results, "correctness.hidden.status"))
    quick_pass = bool_pass(get_nested(results, "correctness.quick.status"))
    smoke_pass = bool_pass(get_nested(results, "correctness.smoke.status"))
    anti_pass = bool_pass(get_nested(results, "anti_cheating.status"))
    speedup_eager = float(get_nested(results, "benchmark.speedup_vs_eager_p50", 0.0) or 0.0)
    speedup_compile = float(get_nested(results, "benchmark.speedup_vs_torch_compile_p50", 0.0) or 0.0)

    first_compile = None
    first_correct = None
    invalid_compiles = 0
    for idx, event in enumerate(trace):
        if str(get_nested(event, "compile.status", "")).lower() == "pass" and first_compile is None:
            first_compile = idx + 1
        if str(get_nested(event, "correctness.hidden.status", "")).lower() == "pass" and first_correct is None:
            first_correct = idx + 1
        if str(event.get("result", "")) == "compile_fail":
            invalid_compiles += 1
    if first_compile is None and compile_pass:
        first_compile = len(trace) or 1
    if first_correct is None and hidden_pass:
        first_correct = len(trace) or 1

    return {
        "run_dir": str(run_dir),
        "group": group,
        "task_id": task.get("task_id"),
        "op_family": task.get("op_family"),
        "op_name": task.get("op_name"),
        "shape": str(task.get("shape")),
        "dtype": task.get("dtype"),
        "seed": run_dir.name.replace("seed", "") if run_dir.name.startswith("seed") else "",
        "final_decision": results.get("final_decision"),
        "compile_success": compile_pass,
        "smoke_pass": smoke_pass,
        "quick_pass": quick_pass,
        "hidden_pass": hidden_pass,
        "anti_cheating_pass": anti_pass,
        "latency_p50_ms": float(get_nested(results, "benchmark.latency_p50_ms", math.nan) or math.nan),
        "latency_p95_ms": float(get_nested(results, "benchmark.latency_p95_ms", math.nan) or math.nan),
        "speedup_vs_eager_p50": speedup_eager,
        "speedup_vs_torch_compile_p50": speedup_compile,
        "correct_and_faster_vs_eager": 1 if hidden_pass and speedup_eager >= 1.0 else 0,
        "correct_and_faster_vs_torch_compile": 1 if hidden_pass and speedup_compile >= 1.0 else 0,
        "iterations": int(get_nested(results, "cost.iterations", len(trace)) or len(trace)),
        "iterations_to_first_compile": first_compile if first_compile is not None else math.nan,
        "iterations_to_first_correct": first_correct if first_correct is not None else math.nan,
        "wall_time_s": float(get_nested(results, "cost.wall_time_s", 0.0) or 0.0),
        "gpu_benchmark_runs": int(get_nested(results, "cost.gpu_benchmark_runs", 0) or 0),
        "invalid_compile_attempts": invalid_compiles,
        "retrieved_capsule_count": len(context.get("retrieved_capsules", [])) if isinstance(context.get("retrieved_capsules"), list) else 0,
        "context_packet_tokens_proxy": len(str(context).split()),
    }


def median(values: list[Any]) -> float:
    nums = []
    for value in values:
        try:
            x = float(value)
            if not math.isnan(x):
                nums.append(x)
        except (TypeError, ValueError):
            pass
    return statistics.median(nums) if nums else math.nan


def rate(values: list[Any]) -> float:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else math.nan


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["group"]), str(row["op_family"]))].append(row)
    out = []
    for (group, op_family), items in sorted(groups.items()):
        out.append({
            "group": group,
            "op_family": op_family,
            "runs": len(items),
            "compile_success_rate": rate([r["compile_success"] for r in items]),
            "hidden_correctness_pass_rate": rate([r["hidden_pass"] for r in items]),
            "anti_cheating_pass_rate": rate([r["anti_cheating_pass"] for r in items]),
            "correct_and_faster_rate_vs_eager": rate([r["correct_and_faster_vs_eager"] for r in items]),
            "correct_and_faster_rate_vs_torch_compile": rate([r["correct_and_faster_vs_torch_compile"] for r in items]),
            "median_speedup_vs_eager_p50": median([r["speedup_vs_eager_p50"] for r in items if r["hidden_pass"]]),
            "median_speedup_vs_torch_compile_p50": median([r["speedup_vs_torch_compile_p50"] for r in items if r["hidden_pass"]]),
            "median_iterations_to_first_correct": median([r["iterations_to_first_correct"] for r in items]),
            "median_wall_time_s": median([r["wall_time_s"] for r in items]),
            "median_invalid_compile_attempts": median([r["invalid_compile_attempts"] for r in items]),
            "median_context_packet_tokens_proxy": median([r["context_packet_tokens_proxy"] for r in items]),
        })
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(("" if math.isnan(value) else f"{value:.4g}") if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="Markdown report path.")
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--metrics", default=None, help="Accepted for protocol traceability.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for root in args.runs:
        for results_path in sorted(Path(root).rglob("results.json")):
            row = collect_run(results_path.parent)
            if row:
                rows.append(row)
    summaries = summarize(rows)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    json_path = Path(args.json_out) if args.json_out else out_path.with_suffix(".json")
    write_json(json_path, {"runs": rows, "summary": summaries})
    columns = ["group", "op_family", "runs", "compile_success_rate", "hidden_correctness_pass_rate", "correct_and_faster_rate_vs_torch_compile", "median_speedup_vs_torch_compile_p50", "median_iterations_to_first_correct", "median_wall_time_s", "median_invalid_compile_attempts"]
    report = ["# H20 MVP Experiment Report", "", f"- run_records: {len(rows)}", f"- csv: `{csv_path}`", f"- json: `{json_path}`", "", "## Group summary", "", markdown_table(summaries, columns) if summaries else "No results found.", "", "## Interpretation checklist", "", "- Compare A3 against A0/A1/A2 under the same task, seed, and budget.", "- Treat hidden correctness as a hard gate before reading performance metrics.", "- Check whether A3 reduces iterations, wall time, invalid compile attempts, and context cost.", "- For self-evolution, compare A3(v1) against A3(v0) on Round-2 held-out tasks."]
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
