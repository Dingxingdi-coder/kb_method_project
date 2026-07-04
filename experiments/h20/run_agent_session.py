#!/usr/bin/env python3
"""Run one fixed-budget Agent session workspace.

This wrapper is Agent-agnostic. It prepares task/context files, optionally invokes an external
Agent command, and then executes the fixed harness. The external command may read/write files in
`--workspace`; it must not modify harness or hidden tests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def copy_or_write_placeholder_candidate(workspace: Path, task: dict[str, Any]) -> None:
    candidate = workspace / "candidate.py"
    if candidate.exists():
        return
    candidate.write_text(
        """# Candidate placeholder. Replace this file with Agent-generated code.\n\n"
        "def run(*inputs, task=None):\n"
        "    raise NotImplementedError('candidate.py has not been implemented yet')\n"
        """,
        encoding="utf-8",
    )
    (workspace / "notes.md").write_text(f"# Notes\n\nTask: `{task.get('name', 'unknown')}`\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--group", default="A3_ecc_kb")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--phase", default="generate")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--agent-command", default=None, help="Optional shell command run inside the workspace each iteration.")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    workspace = Path(args.workspace)
    if workspace.exists() and args.force:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    task = load_json(args.task)
    shutil.copyfile(args.task, workspace / "task.json")
    shutil.copyfile(args.backend, workspace / "backend.json")

    retrieve_cmd = [
        sys.executable,
        str(root / "tools" / "retrieve_context.py"),
        "--task",
        str(workspace / "task.json"),
        "--backend",
        str(workspace / "backend.json"),
        "--group",
        args.group,
        "--phase",
        args.phase,
        "--kb-version",
        args.kb_version,
        "--out",
        str(workspace / "context_packet.json"),
    ]
    subprocess.run(retrieve_cmd, check=True)
    copy_or_write_placeholder_candidate(workspace, task)

    started = time.time()
    trace = workspace / "trace.jsonl"
    final_code = 1
    for iteration in range(args.max_iters):
        append_jsonl(trace, {"iteration": iteration, "phase": args.phase, "action_summary": "start_iteration", "result": "needs_more_evidence"})
        if args.agent_command:
            env = os.environ.copy()
            env.update({"ECC_WORKSPACE": str(workspace), "ECC_TASK": str(workspace / "task.json"), "ECC_CONTEXT": str(workspace / "context_packet.json"), "ECC_ITERATION": str(iteration)})
            proc = subprocess.run(args.agent_command, cwd=workspace, shell=True, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (workspace / f"agent_iter_{iteration}.log").write_text(proc.stdout, encoding="utf-8")
            append_jsonl(trace, {"iteration": iteration, "phase": args.phase, "action_summary": "agent_command", "result": "keep_candidate" if proc.returncode == 0 else "compile_fail", "diagnosis": proc.stdout[-2000:]})

        harness_cmd = [
            sys.executable,
            str(root / "experiments" / "h20" / "harness.py"),
            "--task",
            str(workspace / "task.json"),
            "--candidate",
            str(workspace / "candidate.py"),
            "--out-dir",
            str(workspace),
            "--seed",
            str(args.seed + iteration),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        proc = subprocess.run(harness_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (workspace / f"harness_iter_{iteration}.log").write_text(proc.stdout, encoding="utf-8")
        final_code = proc.returncode
        try:
            results = load_json(workspace / "results.json")
            hidden = ((results.get("correctness") or {}).get("hidden") or {}).get("status")
            append_jsonl(trace, {"iteration": iteration, "phase": "eval", "action_summary": "run_harness", "result": "keep_candidate" if hidden == "pass" else "correctness_fail", "diagnosis": f"harness_exit={proc.returncode}"})
            if hidden == "pass":
                break
        except Exception:
            append_jsonl(trace, {"iteration": iteration, "phase": "eval", "action_summary": "run_harness", "result": "compile_fail", "diagnosis": proc.stdout[-2000:]})

    if (workspace / "results.json").exists():
        results = load_json(workspace / "results.json")
        results.setdefault("cost", {})["wall_time_s"] = time.time() - started
        results["cost"]["iterations"] = min(args.max_iters, iteration + 1)
        write_json(workspace / "results.json", results)
    print(workspace)
    raise SystemExit(final_code)


if __name__ == "__main__":
    main()
