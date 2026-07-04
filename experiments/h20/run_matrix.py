#!/usr/bin/env python3
"""Run a fixed matrix of groups, tasks, and seeds.

This script is intentionally simple. It serializes runs by default; use `--parallel` only when your
GPU lease policy prevents two benchmark processes from sharing one device.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def task_files(task_dir: Path, limit: int = 0) -> list[Path]:
    files = sorted(p for p in task_dir.glob("*.json") if p.name != "manifest.json")
    return files[:limit] if limit else files


def safe_name(path: Path) -> str:
    return path.stem.replace("/", "_")


def run_one(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--groups", required=True, help="Comma-separated groups, e.g. A0_no_kb,A3_ecc_kb")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids used round-robin as CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--agent-command", default=None)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tasks = task_files(Path(args.tasks), args.limit)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [os.environ.get("CUDA_VISIBLE_DEVICES", "0")]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    idx = 0
    for group in groups:
        for task in tasks:
            for seed in seeds:
                workspace = out_root / group / safe_name(task) / f"seed{seed}"
                cmd = [
                    sys.executable,
                    str(root / "experiments" / "h20" / "run_agent_session.py"),
                    "--task",
                    str(task),
                    "--backend",
                    args.backend,
                    "--group",
                    group,
                    "--kb-version",
                    args.kb_version,
                    "--seed",
                    str(seed),
                    "--workspace",
                    str(workspace),
                    "--max-iters",
                    str(args.max_iters),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--force",
                ]
                if args.agent_command:
                    cmd += ["--agent-command", args.agent_command]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpus[idx % len(gpus)]
                env["TRITON_CACHE_DIR"] = str(workspace / ".triton_cache")
                env["TORCHINDUCTOR_CACHE_DIR"] = str(workspace / ".torchinductor_cache")
                jobs.append((cmd, env, workspace))
                idx += 1

    results = []
    if args.parallel <= 1:
        for cmd, env, workspace in jobs:
            res = run_one(cmd, env)
            results.append({"workspace": str(workspace), **res})
            print(f"{res['returncode']} {workspace}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(run_one, cmd, env): workspace for cmd, env, workspace in jobs}
            for fut in concurrent.futures.as_completed(futs):
                workspace = futs[fut]
                res = fut.result()
                results.append({"workspace": str(workspace), **res})
                print(f"{res['returncode']} {workspace}")

    (out_root / "matrix_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [r for r in results if r["returncode"] not in (0, 2)]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
