#!/usr/bin/env python3
"""Posthoc H20 evaluator for final candidates and Hook snapshots."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, read_jsonl, sha256_file, utc_now, write_json  # noqa: E402


def status_pass(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("status")
    return str(value).lower() in {"pass", "passed", "true", "1"}


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def find_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if (root / "task.json").exists() and (root / "candidate.py").exists():
            run_dirs.add(root)
            continue
        for task_path in root.rglob("task.json"):
            run_dir = task_path.parent
            if (run_dir / "candidate.py").exists():
                run_dirs.add(run_dir)
    return sorted(run_dirs)


def hidden_from_run_sh(run_dir: Path) -> Path | None:
    run_sh = run_dir / "run.sh"
    if not run_sh.exists():
        return None
    match = re.search(r"--hidden-tests\s+([^ \n]+)", run_sh.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        return None
    path = Path(match.group(1).strip("'\""))
    if not path.is_absolute():
        path = run_dir / path
    return path if path.exists() else None


def infer_hidden_path(run_dir: Path, hidden_root: Path | None) -> Path | None:
    path = hidden_from_run_sh(run_dir)
    if path is not None:
        return path
    if hidden_root is None:
        return None
    task = read_json(run_dir / "task.json", default={})
    task_id = task.get("task_id")
    if not task_id:
        return None
    path = hidden_root / f"{task_id}.hidden.json"
    return path if path.exists() else None


def snapshot_metadata(run_dir: Path) -> dict[Path, dict[str, Any]]:
    metrics_dir = run_dir / ".codex_h20_metrics"
    out: dict[Path, dict[str, Any]] = {}
    for event in read_jsonl(metrics_dir / "trace.jsonl"):
        if event.get("event") != "candidate_snapshot" or not event.get("snapshot_file"):
            continue
        path = (metrics_dir / str(event["snapshot_file"])).resolve()
        out.setdefault(path, event)
    return out


def candidates_for_run(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [{"kind": "final", "candidate": run_dir / "candidate.py", "snapshot_file": "", "elapsed_from_agent_s": None}]
    meta = snapshot_metadata(run_dir)
    metrics_dir = run_dir / ".codex_h20_metrics"
    for path in sorted((metrics_dir / "candidates").glob("*.py")) if metrics_dir.exists() else []:
        event = meta.get(path.resolve(), {})
        candidates.append(
            {
                "kind": "snapshot",
                "candidate": path,
                "snapshot_file": str(path.relative_to(run_dir)),
                "elapsed_from_agent_s": event.get("elapsed_from_agent_s"),
                "candidate_seq": event.get("candidate_seq"),
            }
        )
    return candidates


def run_harness(args: argparse.Namespace, run_dir: Path, hidden_path: Path | None, candidate: Path, out_dir: Path) -> tuple[int, dict[str, Any], str, str]:
    cmd = [
        args.python,
        str(args.harness.resolve()),
        "--stage",
        "full",
        "--task",
        str(run_dir / "task.json"),
        "--candidate",
        str(candidate),
        "--out-dir",
        str(out_dir),
        "--run",
        str(args.run),
        "--device",
        args.device,
    ]
    if args.require_cuda:
        cmd.append("--require-cuda")
    if hidden_path is not None:
        cmd.extend(["--hidden-tests", str(hidden_path)])
    if args.warmup is not None:
        cmd.extend(["--warmup", str(args.warmup)])
    if args.repeats is not None:
        cmd.extend(["--repeats", str(args.repeats)])
    proc = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout, check=False)
    results = read_json(out_dir / "results.json", default={})
    return proc.returncode, results, proc.stdout[-4000:], proc.stderr[-4000:]


def candidate_row(run_dir: Path, info: dict[str, Any], rc: int, results: dict[str, Any], stdout_tail: str, stderr_tail: str, out_dir: Path) -> dict[str, Any]:
    latency = json_metric(get_nested(results, "benchmark.latency_p50_ms", math.nan))
    return {
        "run_dir": str(run_dir),
        "task_id": results.get("task_id", read_json(run_dir / "task.json", default={}).get("task_id", "")),
        "candidate_kind": info.get("kind"),
        "candidate_path": str(info["candidate"]),
        "snapshot_file": info.get("snapshot_file", ""),
        "candidate_seq": info.get("candidate_seq", ""),
        "elapsed_from_agent_s": info.get("elapsed_from_agent_s"),
        "candidate_hash": results.get("candidate_hash") or sha256_file(info["candidate"]),
        "returncode": rc,
        "compile_success": int(status_pass(get_nested(results, "compile.status"))),
        "smoke_pass": int(status_pass(get_nested(results, "correctness.smoke.status"))),
        "quick_pass": int(status_pass(get_nested(results, "correctness.quick.status"))),
        "hidden_pass": int(status_pass(get_nested(results, "correctness.hidden.status"))),
        "latency_p50_ms": latency,
        "latency_p95_ms": json_metric(get_nested(results, "benchmark.latency_p95_ms", math.nan)),
        "speedup_vs_eager_p50": json_metric(get_nested(results, "benchmark.speedup_vs_eager_p50", 0.0)),
        "speedup_vs_torch_compile_p50": json_metric(get_nested(results, "benchmark.speedup_vs_torch_compile_p50", 0.0)),
        "final_decision": results.get("final_decision", ""),
        "diagnosis": results.get("diagnosis", ""),
        "eval_out_dir": str(out_dir),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def json_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def valid_latency(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("latency_p50_ms", math.nan))
    except (TypeError, ValueError):
        return math.nan
    return value


def best_correct(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    correct = [row for row in rows if int(row.get("hidden_pass", 0)) == 1 and not math.isnan(valid_latency(row))]
    if not correct:
        return None
    return min(correct, key=valid_latency)


def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "candidate_kind": row.get("candidate_kind"),
        "snapshot_file": row.get("snapshot_file"),
        "elapsed_from_agent_s": row.get("elapsed_from_agent_s"),
        "candidate_hash": row.get("candidate_hash"),
        "hidden_pass": row.get("hidden_pass"),
        "latency_p50_ms": row.get("latency_p50_ms"),
        "speedup_vs_eager_p50": row.get("speedup_vs_eager_p50"),
        "eval_out_dir": row.get("eval_out_dir"),
    }


def summarize_run(rows: list[dict[str, Any]], timepoints: list[float]) -> dict[str, Any]:
    final = next((row for row in rows if row.get("candidate_kind") == "final"), None)
    snapshots = [row for row in rows if row.get("candidate_kind") == "snapshot"]
    anytime: dict[str, Any] = {}
    for timepoint in timepoints:
        eligible = []
        for row in snapshots:
            elapsed = row.get("elapsed_from_agent_s")
            try:
                if elapsed is not None and float(elapsed) <= timepoint:
                    eligible.append(row)
            except (TypeError, ValueError):
                pass
        anytime[str(int(timepoint) if float(timepoint).is_integer() else timepoint)] = compact(best_correct(eligible))
    return {
        "run_dir": rows[0]["run_dir"] if rows else "",
        "task_id": rows[0]["task_id"] if rows else "",
        "official_final": compact(final),
        "oracle_best": compact(best_correct(rows)),
        "anytime_best": anytime,
        "candidate_count": len(rows),
        "snapshot_count": len(snapshots),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key in rows[0].keys() if key not in {"stdout_tail", "stderr_tail"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or roots containing run directories.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--harness", type=Path, default=Path("experiments/h20/harness.py"))
    parser.add_argument("--hidden-root", type=Path, default=Path("artifacts/h20/tasks_expanded_pilot/_hidden"))
    parser.add_argument("--eval-dir-name", default="posthoc_eval")
    parser.add_argument("--timepoints", default="120,240,480", help="Comma-separated anytime cutoffs in seconds.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", type=int, default=None, help="Run index; also used for harness input reproducibility.")
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()
    if args.run is None:
        args.run = 0 if args.seed is None else args.seed

    timepoints = [float(x) for x in args.timepoints.split(",") if x.strip()]
    run_dirs = find_run_dirs([Path(root) for root in args.runs])
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        hidden_path = infer_hidden_path(run_dir, args.hidden_root.resolve() if args.hidden_root else None)
        run_rows = []
        for info in candidates_for_run(run_dir):
            label = "final" if info["kind"] == "final" else Path(str(info["snapshot_file"])).stem
            out_dir = run_dir / args.eval_dir_name / label
            rc, results, stdout_tail, stderr_tail = run_harness(args, run_dir, hidden_path, info["candidate"], out_dir)
            row = candidate_row(run_dir, info, rc, results, stdout_tail, stderr_tail, out_dir)
            rows.append(row)
            run_rows.append(row)
        summaries.append(summarize_run(run_rows, timepoints))

    report = {
        "schema_version": "0.1",
        "timestamp": utc_now(),
        "run_count": len(run_dirs),
        "candidate_count": len(rows),
        "timepoints_s": timepoints,
        "summaries": summaries,
        "rows": rows,
    }
    write_json(args.out_json, report)
    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    print(f"evaluated {len(rows)} candidates from {len(run_dirs)} runs")
    print(f"wrote {args.out_json}")
    if args.out_csv:
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
