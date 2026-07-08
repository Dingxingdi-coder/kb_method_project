#!/usr/bin/env python3
"""Record externally observed Coding Agent session timing for one workspace."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, utc_now, write_json  # noqa: E402


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def elapsed_s(started_at: str | None, completed_at: str | None) -> float | None:
    started = parse_time(started_at)
    completed = parse_time(completed_at)
    if not started or not completed:
        return None
    return max(0.0, (completed - started).total_seconds())


def infer_group(workspace: Path) -> str:
    for part in workspace.resolve().parts:
        if part.startswith("A"):
            return part
    return "unknown"


def infer_task_id(workspace: Path) -> str:
    task_path = workspace / "task.json"
    task = read_json(task_path, default={}) if task_path.exists() else {}
    return str(task.get("task_id") or workspace.parent.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--event", choices=["start", "complete"], required=True)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--agent-nickname", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--final-report", default=None)
    parser.add_argument("--metrics-file", default="agent_session_metrics.json")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    path = workspace / args.metrics_file
    metrics: dict[str, Any] = read_json(path, default={}) if path.exists() else {}
    metrics.setdefault("schema_version", "0.1")
    metrics.setdefault("workspace", str(workspace))
    metrics.setdefault("group", infer_group(workspace))
    metrics.setdefault("task_id", infer_task_id(workspace))

    if args.agent_id:
        metrics["agent_id"] = args.agent_id
    if args.agent_nickname:
        metrics["agent_nickname"] = args.agent_nickname
    if args.cuda_visible_devices is not None:
        metrics["cuda_visible_devices"] = args.cuda_visible_devices

    now = utc_now()
    if args.event == "start":
        metrics["agent_started_at"] = now
        metrics["agent_completed_at"] = None
        metrics["agent_wall_time_s"] = None
        metrics["status"] = "running"
    else:
        metrics.setdefault("agent_started_at", now)
        metrics["agent_completed_at"] = now
        metrics["agent_wall_time_s"] = elapsed_s(metrics.get("agent_started_at"), metrics.get("agent_completed_at"))
        metrics["status"] = args.status or "completed"
        if args.final_report is not None:
            metrics["final_report"] = args.final_report

    write_json(path, metrics)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
