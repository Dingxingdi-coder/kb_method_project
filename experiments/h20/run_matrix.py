#!/usr/bin/env python3
"""Prepare or evaluate a matrix of H20 MVP tasks, groups, and runs.

By default this script prepares workspaces. Pass --run-harness to evaluate the
current candidate.py files in those workspaces.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, safe_id, write_json  # noqa: E402


def discover_tasks(task_dir: Path) -> list[Path]:
    return [p for p in sorted(task_dir.glob("*.json")) if p.name != "manifest.json" and not p.name.endswith(".hidden.json")]


def apply_filter(tasks: list[Path], task_filter: str | None) -> list[Path]:
    if not task_filter:
        return tasks
    if task_filter.startswith("pilot"):
        try:
            return tasks[: int(task_filter.replace("pilot", ""))]
        except ValueError:
            return tasks[:12]
    terms = [t.strip().lower() for t in task_filter.split(",") if t.strip()]
    return [t for t in tasks if any(term in t.name.lower() for term in terms)]


def hidden_for(task: Path) -> Path | None:
    hidden = task.parent / "_hidden" / f"{task.stem}.hidden.json"
    return hidden if hidden.exists() else None


def run_one(repo_root: Path, runner: Path, job: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    cmd = [sys.executable, str(runner), "--group", job["group"], "--task", str(job["task"]), "--backend", str(job["backend"]), "--kb-version", job["kb_version"], "--run", str(job["run"]), "--out", str(job["out_dir"]), "--phase", job["phase"], "--iteration", str(job["iteration"])]
    if job.get("hidden"):
        cmd.extend(["--hidden-tests", str(job["hidden"])])
    if job.get("run_harness"):
        cmd.append("--run-harness")
    if job.get("warmup") is not None:
        cmd.extend(["--warmup", str(job["warmup"])])
    if job.get("repeats") is not None:
        cmd.extend(["--repeats", str(job["repeats"])])
    if job.get("conda_env") is not None:
        cmd.extend(["--conda-env", str(job["conda_env"])])
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"job": {k: str(v) for k, v in job.items()}, "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--groups", required=True, help="Comma-separated group names.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--runs", default=None, help="Comma-separated run indices.")
    parser.add_argument("--seeds", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--run-harness", action="store_true")
    parser.add_argument("--phase", default="generate")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--conda-env", default="op_kb_dxd", help="Conda environment activated by generated run.sh. Use an empty value to disable activation.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    runner = repo_root / "experiments" / "h20" / "run_agent_session.py"
    tasks = apply_filter(discover_tasks(Path(args.tasks)), args.task_filter)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    run_arg = args.runs if args.runs is not None else (args.seeds if args.seeds is not None else "0")
    runs = [int(s.strip()) for s in run_arg.split(",") if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    idx = 0
    for task in tasks:
        task_data = read_json(task)
        task_id = task_data.get("task_id", task.stem)
        for group in groups:
            for run in runs:
                gpu = gpus[idx % len(gpus)]; idx += 1
                jobs.append({"task": task.resolve(), "hidden": hidden_for(task), "group": group, "run": run, "gpu": gpu, "backend": Path(args.backend).resolve(), "kb_version": args.kb_version, "phase": args.phase, "iteration": args.iteration, "run_harness": args.run_harness, "warmup": args.warmup, "repeats": args.repeats, "conda_env": args.conda_env, "out_dir": (out_root / group / safe_id(str(task_id)) / f"run{run}").resolve()})

    write_json(out_root / "matrix_manifest.json", {"jobs": [{k: str(v) for k, v in job.items()} for job in jobs]})
    print(f"scheduled jobs={len(jobs)} tasks={len(tasks)} groups={groups} runs={runs}")
    results = []
    for job in jobs:
        result = run_one(repo_root, runner, job)
        results.append(result)
        print(f"done rc={result['returncode']} {job['out_dir']}")
    write_json(out_root / "matrix_results.json", {"results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
