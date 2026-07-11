#!/usr/bin/env python3
"""Aggregate H20 MVP results into CSV, JSON, and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
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


def optional_bool_pass(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("status")
    if str(value).lower() in ("not_run", "skipped", "none", "null", ""):
        return None
    return bool_pass(value)


def load_context(run_dir: Path) -> dict[str, Any]:
    packets_dir = run_dir / "context_packets"
    packet_paths = sorted(packets_dir.glob("*.json")) if packets_dir.exists() else []
    packets = [read_json(path, default={}) for path in packet_paths]
    packets = [packet for packet in packets if isinstance(packet, dict)]
    if not packets:
        legacy_path = run_dir / "context_packet.json"
        return read_json(legacy_path, default={}) if legacy_path.exists() else {}

    merged: dict[str, Any] = {
        "group": packets[0].get("group"),
        "retrieval_mode": ",".join(sorted({str(p.get("retrieval_mode", "unknown")) for p in packets})),
        "retrieved_items": [],
        "retrieved_item_ids": [],
        "retrieved_capsules": [],
        "source_corpus_version": ",".join(sorted({str(p.get("source_corpus_version", "")) for p in packets if p.get("source_corpus_version")})),
        "raw_corpus_index_version": ",".join(sorted({str(p.get("raw_corpus_index_version", "")) for p in packets if p.get("raw_corpus_index_version")})),
        "kb_version": ",".join(sorted({str(p.get("kb_version", "")) for p in packets if p.get("kb_version")})),
        "kb_plain_rag_index_version": ",".join(sorted({str(p.get("kb_plain_rag_index_version", "")) for p in packets if p.get("kb_plain_rag_index_version")})),
        "kb_vector_index_version": ",".join(sorted({str(p.get("kb_vector_index_version", "")) for p in packets if p.get("kb_vector_index_version")})),
        "ecc_kb_index_version": ",".join(sorted({str(p.get("ecc_kb_index_version", "")) for p in packets if p.get("ecc_kb_index_version")})),
        "vector_index_version": ",".join(sorted({str(p.get("vector_index_version", "")) for p in packets if p.get("vector_index_version")})),
        "vector_index_hash": ",".join(sorted({str(p.get("vector_index_hash", "")) for p in packets if p.get("vector_index_hash")})),
        "embedding_model_id": ",".join(sorted({str(p.get("embedding_model_id", "")) for p in packets if p.get("embedding_model_id")})),
        "_packets": packets,
    }
    seen_ids: set[str] = set()
    for packet in packets:
        for item in packet.get("retrieved_items", []):
            if isinstance(item, dict):
                item_id = str(item.get("id", ""))
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                merged["retrieved_items"].append(item)
        for item_id in packet.get("retrieved_item_ids", []):
            item_id = str(item_id)
            if item_id and item_id not in merged["retrieved_item_ids"]:
                merged["retrieved_item_ids"].append(item_id)
        for capsule_id in packet.get("retrieved_capsules", []):
            capsule_id = str(capsule_id)
            if capsule_id and capsule_id not in merged["retrieved_capsules"]:
                merged["retrieved_capsules"].append(capsule_id)
    return merged


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    results_path = run_dir / "results.json"
    task_path = run_dir / "task.json"
    trace_path = run_dir / "trace.jsonl"
    agent_metrics_path = run_dir / "agent_session_metrics.json"
    if not results_path.exists() or not task_path.exists():
        return None
    results = read_json(results_path)
    task = read_json(task_path)
    context = load_context(run_dir)
    trace = read_jsonl(trace_path) if trace_path.exists() else []
    agent_metrics = read_json(agent_metrics_path, default={}) if agent_metrics_path.exists() else {}
    group = context.get("group") or next((p for p in run_dir.parts if p.startswith("A")), "unknown")

    compile_pass = bool_pass(get_nested(results, "compile.status"))
    hidden_pass = bool_pass(get_nested(results, "correctness.hidden.status"))
    quick_pass = bool_pass(get_nested(results, "correctness.quick.status"))
    smoke_pass = bool_pass(get_nested(results, "correctness.smoke.status"))
    anti_pass = optional_bool_pass(get_nested(results, "anti_cheating.status"))
    speedup_eager = float(get_nested(results, "benchmark.speedup_vs_eager_p50", 0.0) or 0.0)
    speedup_compile = float(get_nested(results, "benchmark.speedup_vs_torch_compile_p50", 0.0) or 0.0)
    retrieved_items = context.get("retrieved_items", [])
    if not isinstance(retrieved_items, list):
        retrieved_items = []
    retrieved_item_ids = context.get("retrieved_item_ids", [])
    if not isinstance(retrieved_item_ids, list):
        retrieved_item_ids = []
    if not retrieved_item_ids and isinstance(context.get("retrieved_capsules"), list):
        retrieved_item_ids = context.get("retrieved_capsules", [])
    source_type_distribution = Counter(
        str(item.get("source_type", "unknown")) for item in retrieved_items if isinstance(item, dict)
    )
    raw_corpus_count = sum(source_type_distribution.get(k, 0) for k in ("raw_corpus", "raw_archive"))
    kb_plain_count = source_type_distribution.get("kb_unit", 0)
    ecc_capsule_count = source_type_distribution.get("ecc_capsule", 0)
    if not ecc_capsule_count and isinstance(context.get("retrieved_capsules"), list):
        ecc_capsule_count = len(context.get("retrieved_capsules", []))

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

    failure_reason = ""
    if not compile_pass:
        failure_reason = str(get_nested(results, "compile.reason", "") or "compile failed")
    elif not smoke_pass:
        failure_reason = str(get_nested(results, "correctness.smoke.tests.0.reason", "") or "smoke correctness failed")
    elif not quick_pass:
        failure_reason = "quick correctness failed"
    elif not hidden_pass:
        failure_reason = str(results.get("diagnosis") or "hidden correctness failed")

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
        "harness_wall_time_s": float(get_nested(results, "cost.wall_time_s", 0.0) or 0.0),
        "agent_wall_time_s": float(agent_metrics.get("agent_wall_time_s", math.nan) or math.nan),
        "agent_started_at": agent_metrics.get("agent_started_at", ""),
        "agent_completed_at": agent_metrics.get("agent_completed_at", ""),
        "agent_status": agent_metrics.get("status", ""),
        "gpu_id": agent_metrics.get("gpu_id", agent_metrics.get("cuda_visible_devices", "")),
        "failure_reason": failure_reason,
        "gpu_benchmark_runs": int(get_nested(results, "cost.gpu_benchmark_runs", 0) or 0),
        "invalid_compile_attempts": invalid_compiles,
        "retrieved_context_length": len(str(context)),
        "retrieval_mode": context.get("retrieval_mode", "unknown"),
        "source_corpus_version": context.get("source_corpus_version", ""),
        "raw_corpus_index_version": context.get("raw_corpus_index_version", ""),
        "kb_version": context.get("kb_version", ""),
        "kb_plain_rag_index_version": context.get("kb_plain_rag_index_version", ""),
        "kb_vector_index_version": context.get("kb_vector_index_version", ""),
        "ecc_kb_index_version": context.get("ecc_kb_index_version", ""),
        "vector_index_version": context.get("vector_index_version", ""),
        "vector_index_hash": context.get("vector_index_hash", ""),
        "embedding_model_id": context.get("embedding_model_id", ""),
        "retrieved_item_count": len(retrieved_item_ids),
        "retrieved_item_ids": ",".join(str(x) for x in retrieved_item_ids),
        "retrieved_source_type_distribution": dict(source_type_distribution),
        "raw_corpus_retrieved_chunk_count": raw_corpus_count,
        "kb_plain_rag_retrieved_unit_count": kb_plain_count,
        "retrieved_capsule_count": ecc_capsule_count,
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
            "median_harness_wall_time_s": median([r["harness_wall_time_s"] for r in items]),
            "median_agent_wall_time_s": median([r["agent_wall_time_s"] for r in items]),
            "median_invalid_compile_attempts": median([r["invalid_compile_attempts"] for r in items]),
            "median_retrieved_context_length": median([r["retrieved_context_length"] for r in items]),
            "median_retrieved_item_count": median([r["retrieved_item_count"] for r in items]),
            "median_raw_corpus_retrieved_chunk_count": median([r["raw_corpus_retrieved_chunk_count"] for r in items]),
            "median_kb_plain_rag_retrieved_unit_count": median([r["kb_plain_rag_retrieved_unit_count"] for r in items]),
            "median_retrieved_capsule_count": median([r["retrieved_capsule_count"] for r in items]),
        })
    return out


def summarize_overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group"])].append(row)
    out = []
    for group, items in sorted(groups.items()):
        out.append({
            "group": group,
            "runs": len(items),
            "op_families": ",".join(sorted({str(row["op_family"]) for row in items})),
            "compile_success_rate": rate([r["compile_success"] for r in items]),
            "hidden_correctness_pass_rate": rate([r["hidden_pass"] for r in items]),
            "correct_and_faster_rate_vs_eager": rate([r["correct_and_faster_vs_eager"] for r in items]),
            "correct_and_faster_rate_vs_torch_compile": rate([r["correct_and_faster_vs_torch_compile"] for r in items]),
            "median_latency_p50_ms": median([r["latency_p50_ms"] for r in items if r["hidden_pass"]]),
            "median_latency_p95_ms": median([r["latency_p95_ms"] for r in items if r["hidden_pass"]]),
            "median_speedup_vs_eager_p50": median([r["speedup_vs_eager_p50"] for r in items if r["hidden_pass"]]),
            "median_speedup_vs_torch_compile_p50": median([r["speedup_vs_torch_compile_p50"] for r in items if r["hidden_pass"]]),
            "median_agent_wall_time_s": median([r["agent_wall_time_s"] for r in items]),
            "median_harness_wall_time_s": median([r["harness_wall_time_s"] for r in items]),
            "median_retrieved_context_length": median([r["retrieved_context_length"] for r in items]),
            "median_retrieved_item_count": median([r["retrieved_item_count"] for r in items]),
        })
    return out


def failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row["compile_success"] and row["hidden_pass"]:
            continue
        out.append({
            "group": row["group"],
            "task_id": row["task_id"],
            "op_family": row["op_family"],
            "seed": row["seed"],
            "compile_success": row["compile_success"],
            "hidden_pass": row["hidden_pass"],
            "failure_reason": row["failure_reason"],
            "run_dir": row["run_dir"],
        })
    return out


def run_key(path: Any) -> str:
    return str(Path(str(path)).resolve())


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compact_latency(row: dict[str, Any] | None) -> float:
    return as_float((row or {}).get("latency_p50_ms"))


def compact_speedup(row: dict[str, Any] | None, key: str) -> float:
    return as_float((row or {}).get(key))


def load_official_eval(paths: list[str] | None) -> tuple[dict[str, dict[str, Any]], set[str]]:
    official: dict[str, dict[str, Any]] = {}
    timepoints: set[str] = set()
    if not paths:
        return official, timepoints
    for name in paths:
        report = read_json(Path(name), default={})
        first_correct: dict[str, float] = {}
        for row in report.get("rows", []):
            if int(row.get("hidden_pass", 0) or 0) != 1:
                continue
            elapsed = as_float(row.get("elapsed_from_agent_s"))
            if math.isnan(elapsed):
                continue
            key = run_key(row.get("run_dir", ""))
            first_correct[key] = min(first_correct.get(key, elapsed), elapsed)
        for summary in report.get("summaries", []):
            key = run_key(summary.get("run_dir", ""))
            summary = dict(summary)
            summary["official_time_to_correct_s"] = first_correct.get(key, math.nan)
            official[key] = summary
            anytime = summary.get("anytime_best", {})
            if isinstance(anytime, dict):
                timepoints.update(str(t) for t in anytime)
    return official, timepoints


def load_budget_audit(paths: list[str] | None) -> dict[str, dict[str, Any]]:
    budget: dict[str, dict[str, Any]] = {}
    if not paths:
        return budget
    for name in paths:
        report = read_json(Path(name), default={})
        for row in report.get("runs", []):
            budget[run_key(row.get("workspace", ""))] = row
    return budget


def enrich_with_posthoc(
    rows: list[dict[str, Any]],
    official: dict[str, dict[str, Any]],
    budget: dict[str, dict[str, Any]],
    timepoints: set[str],
) -> None:
    for row in rows:
        key = run_key(row["run_dir"])
        summary = official.get(key, {})
        final = summary.get("official_final") if isinstance(summary.get("official_final"), dict) else None
        oracle = summary.get("oracle_best") if isinstance(summary.get("oracle_best"), dict) else None
        row["official_final_hidden_pass"] = as_int((final or {}).get("hidden_pass"))
        row["official_final_latency_p50_ms"] = compact_latency(final)
        row["official_final_speedup_vs_eager_p50"] = compact_speedup(final, "speedup_vs_eager_p50")
        row["official_oracle_best_latency_p50_ms"] = compact_latency(oracle)
        row["official_oracle_best_speedup_vs_eager_p50"] = compact_speedup(oracle, "speedup_vs_eager_p50")
        row["official_time_to_correct_s"] = as_float(summary.get("official_time_to_correct_s"))
        row["official_time_to_best_s"] = as_float((oracle or {}).get("elapsed_from_agent_s"))
        for timepoint in sorted(timepoints, key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else math.inf):
            anytime = summary.get("anytime_best", {})
            best = anytime.get(timepoint) if isinstance(anytime, dict) else None
            row[f"official_anytime_{timepoint}s_latency_p50_ms"] = compact_latency(best if isinstance(best, dict) else None)
            row[f"official_anytime_{timepoint}s_speedup_vs_eager_p50"] = compact_speedup(best if isinstance(best, dict) else None, "speedup_vs_eager_p50")

        audit = budget.get(key, {})
        row["budget_over_budget"] = int(bool(audit.get("over_budget"))) if audit else None
        row["budget_over_budget_metrics"] = audit.get("over_budget_metrics", "")
        row["budget_candidate_count"] = as_int(audit.get("candidate_count")) if audit else None
        row["budget_compile_attempts"] = as_int(audit.get("compile_attempts")) if audit else None
        row["budget_correctness_runs"] = as_int(audit.get("correctness_runs")) if audit else None
        row["budget_benchmark_runs"] = as_int(audit.get("benchmark_runs")) if audit else None
        row["budget_harness_runs"] = as_int(audit.get("harness_runs")) if audit else None


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
    parser.add_argument("--official-eval", nargs="*", default=None, help="Optional evaluate_official.py JSON report(s) to merge.")
    parser.add_argument("--budget-audit", nargs="*", default=None, help="Optional audit_budgets.py JSON report(s) to merge.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for root in args.runs:
        for results_path in sorted(Path(root).rglob("results.json")):
            row = collect_run(results_path.parent)
            if row:
                rows.append(row)
    official, official_timepoints = load_official_eval(args.official_eval)
    budget = load_budget_audit(args.budget_audit)
    enrich_with_posthoc(rows, official, budget, official_timepoints)
    summaries = summarize(rows)
    overall = summarize_overall(rows)
    failures = failed_rows(rows)
    out_path = Path(args.out)
    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "report.md"
    elif str(args.out).endswith(("/", "\\")):
        out_path = out_path / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    json_path = Path(args.json_out) if args.json_out else out_path.with_suffix(".json")
    write_json(
        json_path,
        {
            "runs": rows,
            "summary": summaries,
            "overall_summary": overall,
            "failures": failures,
            "posthoc_inputs": {
                "official_eval": args.official_eval or [],
                "budget_audit": args.budget_audit or [],
            },
        },
    )
    columns = ["group", "op_family", "runs", "compile_success_rate", "hidden_correctness_pass_rate", "correct_and_faster_rate_vs_torch_compile", "median_speedup_vs_torch_compile_p50", "median_iterations_to_first_correct", "median_agent_wall_time_s", "median_harness_wall_time_s", "median_invalid_compile_attempts", "median_retrieved_context_length", "median_retrieved_item_count"]
    overall_columns = ["group", "runs", "op_families", "compile_success_rate", "hidden_correctness_pass_rate", "median_latency_p50_ms", "median_latency_p95_ms", "median_speedup_vs_eager_p50", "median_speedup_vs_torch_compile_p50", "median_agent_wall_time_s", "median_harness_wall_time_s", "median_retrieved_context_length", "median_retrieved_item_count"]
    failure_columns = ["group", "task_id", "op_family", "seed", "compile_success", "hidden_pass", "failure_reason"]
    official_columns = ["group", "task_id", "op_family", "seed", "official_final_hidden_pass", "official_final_latency_p50_ms", "official_oracle_best_latency_p50_ms", "official_time_to_correct_s", "official_time_to_best_s"]
    budget_columns = ["group", "task_id", "op_family", "seed", "budget_over_budget", "budget_over_budget_metrics", "budget_candidate_count", "budget_compile_attempts", "budget_correctness_runs", "budget_benchmark_runs", "budget_harness_runs"]
    official_rows = [row for row in rows if not math.isnan(as_float(row.get("official_final_latency_p50_ms"))) or not math.isnan(as_float(row.get("official_oracle_best_latency_p50_ms")))]
    budget_rows = [row for row in rows if row.get("budget_over_budget") is not None]
    report = [
        "# H20 MVP Experiment Report",
        "",
        f"- run_records: {len(rows)}",
        f"- failed_or_incorrect_records: {len(failures)}",
        f"- official_eval_records: {len(official_rows)}",
        f"- budget_audit_records: {len(budget_rows)}",
        f"- csv: `{csv_path}`",
        f"- json: `{json_path}`",
        "",
        "## Overall group summary",
        "",
        markdown_table(overall, overall_columns) if overall else "No results found.",
        "",
        "## Group summary by op family",
        "",
        markdown_table(summaries, columns) if summaries else "No results found.",
        "",
        "## Failures",
        "",
        markdown_table(failures, failure_columns) if failures else "No compile or hidden-correctness failures found.",
        "",
        "## Official evaluator",
        "",
        markdown_table(official_rows, official_columns) if official_rows else "No official evaluator report was merged.",
        "",
        "## Budget audit",
        "",
        markdown_table(budget_rows, budget_columns) if budget_rows else "No budget audit report was merged.",
        "",
        "## Interpretation checklist",
        "",
        "- Compare A3 against A0/A1/A2 under the same task, seed, and budget.",
        "- Treat hidden correctness as a hard gate before reading performance metrics.",
        "- Check whether A3 reduces iterations, wall time, invalid compile attempts, and context cost.",
        "- Manually review final `candidate.py` files for target high-level API fallback before claiming legal performance.",
        "- For self-evolution, compare A3(v1) against A3(v0) on Round-2 held-out tasks.",
    ]
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
