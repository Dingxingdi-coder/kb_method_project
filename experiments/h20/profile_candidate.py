#!/usr/bin/env python3
"""Optional post-correctness hardware profiling wrapper for H20 candidates."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, sha256_file, utc_now, write_json  # noqa: E402


def status_at(data: dict[str, Any], path: str, default: str = "not_run") -> str:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return str(cur)


def run_hidden_check(args: argparse.Namespace, out_dir: Path) -> tuple[bool, dict[str, Any], str]:
    cmd = [
        sys.executable,
        str(Path(args.harness).resolve()),
        "--stage",
        "hidden",
        "--task",
        str(Path(args.task).resolve()),
        "--candidate",
        str(Path(args.candidate).resolve()),
        "--out-dir",
        str(out_dir),
        "--run",
        str(args.run),
        "--device",
        args.device,
    ]
    if args.hidden_tests:
        cmd.extend(["--hidden-tests", str(Path(args.hidden_tests).resolve())])
    if args.require_cuda:
        cmd.append("--require-cuda")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    results = read_json(out_dir / "results.json", default={})
    passed = proc.returncode == 0 and status_at(results, "correctness.hidden.status") == "pass"
    return passed, results, proc.stdout


def run_ncu(args: argparse.Namespace, out_dir: Path) -> tuple[str, str, int]:
    ncu = shutil.which(args.ncu_bin)
    if not ncu:
        return "skipped", f"{args.ncu_bin} not found on PATH", 127

    report_base = out_dir / "ncu_report"
    harness_cmd = [
        sys.executable,
        str(Path(args.harness).resolve()),
        "--stage",
        "benchmark",
        "--task",
        str(Path(args.task).resolve()),
        "--candidate",
        str(Path(args.candidate).resolve()),
        "--out-dir",
        str(out_dir),
        "--run",
        str(args.run),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--device",
        args.device,
    ]
    if args.hidden_tests:
        harness_cmd.extend(["--hidden-tests", str(Path(args.hidden_tests).resolve())])
    if args.require_cuda:
        harness_cmd.append("--require-cuda")
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--set",
        args.ncu_set,
        "--export",
        str(report_base),
        "--force-overwrite",
        *harness_cmd,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    status = "pass" if proc.returncode == 0 else "fail"
    return status, proc.stdout, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task.json")
    parser.add_argument("--candidate", default="candidate.py")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--hidden-tests", default=None)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-set", default="speed-of-light")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    task = read_json(args.task, default={})
    candidate_path = Path(args.candidate)

    hidden_ok, hidden_results, hidden_log = run_hidden_check(args, out_dir)
    (out_dir / "hardware_profile_hidden_check.log").write_text(hidden_log, encoding="utf-8")
    if not hidden_ok:
        summary = {
            "schema_version": "0.1",
            "timestamp": utc_now(),
            "profile_source": "ncu",
            "profile_kind": "hardware_profiler",
            "heuristic_profile": False,
            "status": "skipped",
            "reason": "hidden correctness did not pass; hardware profiling is only for post-correctness candidates",
            "task_id": task.get("task_id"),
            "candidate_hash": sha256_file(candidate_path),
            "hidden_status": status_at(hidden_results, "correctness.hidden.status"),
            "wall_time_s": time.time() - started,
        }
        write_json(out_dir / "hardware_profile_summary.json", summary)
        print(summary["reason"])
        return 1

    status, profile_log, returncode = run_ncu(args, out_dir)
    (out_dir / "hardware_profile.log").write_text(profile_log, encoding="utf-8")
    summary = {
        "schema_version": "0.1",
        "timestamp": utc_now(),
        "profile_source": "ncu",
        "profile_kind": "hardware_profiler",
        "heuristic_profile": False,
        "status": status,
        "returncode": returncode,
        "task_id": task.get("task_id"),
        "op_family": task.get("op_family"),
        "candidate_hash": sha256_file(candidate_path),
        "hidden_status": "pass",
        "ncu_set": args.ncu_set,
        "report_base": str(out_dir / "ncu_report"),
        "wall_time_s": time.time() - started,
    }
    if status == "skipped":
        summary["reason"] = profile_log
    elif status == "fail":
        summary["reason"] = "ncu command failed; see hardware_profile.log"
    write_json(out_dir / "hardware_profile_summary.json", summary)
    print(f"hardware profiling: {status}")
    if status != "pass" and profile_log:
        print(profile_log[-2000:])
    return 0 if status in {"pass", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
