#!/usr/bin/env python3
"""Post-hoc budget audit for H20 experiment workspaces.

This script is intentionally observational: it reads existing workspace logs and
writes an aggregate report. It does not modify candidates, workspaces, hooks, or
run control.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, read_jsonl, write_json  # noqa: E402


def infer_run_label(path: Path) -> str:
    name = path.name
    if name.startswith("run"):
        return name.replace("run", "", 1)
    if name.startswith("seed"):
        return name.replace("seed", "", 1)
    return ""


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def get_nested(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def infer_group(workspace: Path, task: dict[str, Any], hook_summary: dict[str, Any], agent_metrics: dict[str, Any]) -> str:
    meta_group = get_nested(hook_summary, "meta.group") or agent_metrics.get("group")
    if meta_group:
        return str(meta_group)
    for part in workspace.resolve().parts:
        if part.startswith("A"):
            return part
    return str(task.get("group") or "unknown")


def max_agent_wall_time(hook_summary: dict[str, Any], agent_metrics: dict[str, Any]) -> float | None:
    values: list[float] = []
    for agent in (hook_summary.get("agents") or {}).values():
        if isinstance(agent, dict):
            value = as_number(agent.get("wall_time_s"))
            if value is not None:
                values.append(value)
    value = as_number(agent_metrics.get("agent_wall_time_s"))
    if value is not None:
        values.append(value)
    return max(values) if values else None


def total_from_hook(hook_summary: dict[str, Any], key: str) -> int | None:
    value = as_number(get_nested(hook_summary, f"totals.{key}"))
    if value is not None:
        return int(value)
    values: list[int] = []
    for agent in (hook_summary.get("agents") or {}).values():
        if isinstance(agent, dict):
            agent_value = as_number(agent.get(key))
            if agent_value is not None:
                values.append(int(agent_value))
    return sum(values) if values else None


def trace_harness_runs(trace: list[dict[str, Any]]) -> int | None:
    if not trace:
        return None
    count = 0
    for event in trace:
        if isinstance(event, dict) and (
            event.get("compile") is not None
            or event.get("correctness") is not None
            or event.get("benchmark") is not None
            or str(event.get("action_summary", "")).startswith("harness_rc=")
        ):
            count += 1
    return count


def compare_budget(name: str, value: float | int | None, limit: float | int | None) -> dict[str, Any]:
    if limit is None:
        return {"metric": name, "value": value, "limit": None, "status": "not_configured"}
    if value is None:
        return {"metric": name, "value": None, "limit": limit, "status": "missing"}
    return {
        "metric": name,
        "value": value,
        "limit": limit,
        "status": "over" if float(value) > float(limit) else "within",
    }


def collect_workspace(workspace: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    task_path = workspace / "task.json"
    candidate_path = workspace / "candidate.py"
    if not task_path.exists() or not candidate_path.exists():
        return None

    task = read_json(task_path, default={})
    results = read_json(workspace / "results.json", default={}) if (workspace / "results.json").exists() else {}
    trace = read_jsonl(workspace / "trace.jsonl") if (workspace / "trace.jsonl").exists() else []
    agent_metrics = read_json(workspace / "agent_session_metrics.json", default={}) if (workspace / "agent_session_metrics.json").exists() else {}
    hook_summary_path = workspace / ".codex_h20_metrics" / "summary.json"
    hook_summary = read_json(hook_summary_path, default={}) if hook_summary_path.exists() else {}

    candidate_count = total_from_hook(hook_summary, "unique_candidate_count")
    if candidate_count is None:
        candidate_count = int(as_number(hook_summary.get("candidate_count")) or 0) if hook_summary else None

    harness_runs = total_from_hook(hook_summary, "harness_runs")
    if harness_runs is None:
        harness_runs = trace_harness_runs(trace)

    benchmark_runs = total_from_hook(hook_summary, "benchmark_runs")
    if benchmark_runs is None:
        benchmark_runs = int(as_number(get_nested(results, "cost.gpu_benchmark_runs")) or 0) if results else None

    values: dict[str, float | int | None] = {
        "agent_wall_time_s": max_agent_wall_time(hook_summary, agent_metrics),
        "candidate_count": candidate_count,
        "compile_attempts": total_from_hook(hook_summary, "compile_attempts"),
        "correctness_runs": total_from_hook(hook_summary, "correctness_runs"),
        "benchmark_runs": benchmark_runs,
        "harness_runs": harness_runs,
    }
    budgets = {
        "agent_wall_time_s": args.max_agent_wall_time_s,
        "candidate_count": args.max_candidates,
        "compile_attempts": args.max_compile_attempts,
        "correctness_runs": args.max_correctness_runs,
        "benchmark_runs": args.max_benchmark_runs,
        "harness_runs": args.max_harness_runs,
    }
    checks = [compare_budget(name, values[name], limit) for name, limit in budgets.items()]
    over = [check["metric"] for check in checks if check["status"] == "over"]
    missing = [check["metric"] for check in checks if check["status"] == "missing"]

    return {
        "workspace": str(workspace),
        "group": infer_group(workspace, task, hook_summary, agent_metrics),
        "task_id": task.get("task_id"),
        "op_family": task.get("op_family"),
        "run": infer_run_label(workspace),
        **values,
        "over_budget": bool(over),
        "over_budget_metrics": ",".join(over),
        "missing_budget_metrics": ",".join(missing),
        "checks": checks,
        "sources": {
            "hook_summary": str(hook_summary_path) if hook_summary_path.exists() else "",
            "agent_session_metrics": str(workspace / "agent_session_metrics.json") if (workspace / "agent_session_metrics.json").exists() else "",
            "results": str(workspace / "results.json") if (workspace / "results.json").exists() else "",
            "trace": str(workspace / "trace.jsonl") if (workspace / "trace.jsonl").exists() else "",
        },
    }


def find_workspaces(roots: list[str]) -> list[Path]:
    workspaces: set[Path] = set()
    for root_name in roots:
        root = Path(root_name).resolve()
        if (root / "task.json").exists() and (root / "candidate.py").exists():
            workspaces.add(root)
            continue
        for task_path in root.rglob("task.json"):
            workspace = task_path.parent
            if (workspace / "candidate.py").exists():
                workspaces.add(workspace)
    return sorted(workspaces)


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"checks", "sources"}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc audit of H20 workspace budget usage.")
    parser.add_argument("--runs", nargs="+", required=True, help="Workspace directories or roots containing workspaces.")
    parser.add_argument("--out", required=True, help="JSON audit report path.")
    parser.add_argument("--csv-out", default=None, help="Optional CSV audit report path.")
    parser.add_argument("--max-agent-wall-time-s", type=float, default=480.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-compile-attempts", type=int, default=None)
    parser.add_argument("--max-correctness-runs", type=int, default=None)
    parser.add_argument("--max-benchmark-runs", type=int, default=None)
    parser.add_argument("--max-harness-runs", type=int, default=None)
    args = parser.parse_args()

    rows = [row for workspace in find_workspaces(args.runs) if (row := collect_workspace(workspace, args))]
    report = {
        "schema_version": "0.1",
        "mode": "post_hoc_observational_only",
        "workspace_count": len(rows),
        "over_budget_count": sum(1 for row in rows if row["over_budget"]),
        "missing_metric_count": sum(1 for row in rows if row["missing_budget_metrics"]),
        "budgets": {
            "max_agent_wall_time_s": args.max_agent_wall_time_s,
            "max_candidates": args.max_candidates,
            "max_compile_attempts": args.max_compile_attempts,
            "max_correctness_runs": args.max_correctness_runs,
            "max_benchmark_runs": args.max_benchmark_runs,
            "max_harness_runs": args.max_harness_runs,
        },
        "runs": rows,
        "notes": [
            "This audit reads existing logs after the experiment.",
            "It does not enforce budgets, stop agents, edit candidates, or modify workspaces.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, report)

    csv_path = Path(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    if rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_row(rows[0]).keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_row(row) for row in rows)

    print(f"wrote {out_path}")
    print(f"workspaces={len(rows)} over_budget={report['over_budget_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
