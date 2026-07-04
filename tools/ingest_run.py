#!/usr/bin/env python3
"""Convert one harness workspace into an ECC-KB evolution record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: str | Path | None, default: Any = None) -> Any:
    if not path:
        return default
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_trace(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                rows.append({"raw": line, "result": "needs_more_evidence"})
    return rows


def final_decision(results: dict[str, Any]) -> str:
    if not results:
        return "NEED_MORE_EVIDENCE"
    if results.get("cheating_detected"):
        return "FAIL"
    compile_status = (results.get("compile") or {}).get("status")
    hidden = ((results.get("correctness") or {}).get("hidden") or {}).get("status") or (results.get("correctness") or {}).get("hidden")
    perf = results.get("performance") or results.get("benchmark") or {}
    if compile_status == "fail":
        return "FAIL"
    if hidden == "pass":
        speedup = perf.get("speedup_vs_torch_compile_p50") or perf.get("speedup_vs_eager_p50") or 0
        try:
            return "KEEP" if float(speedup) >= 1.0 else "DISCARD"
        except Exception:
            return "KEEP"
    if hidden == "fail":
        return "FAIL"
    return "NEED_MORE_EVIDENCE"


def promotion_candidates(results: dict[str, Any]) -> list[dict[str, Any]]:
    decision = final_decision(results)
    candidates: list[dict[str, Any]] = []
    if decision == "KEEP":
        candidates.append(
            {
                "candidate_unit_id": f"auto_{results.get('run_id', 'run')}_perf_motif",
                "reason": "Candidate passed hidden correctness and met a non-regression performance criterion.",
                "required_gates": ["schema", "hidden_correctness", "benchmark_repeat", "reproducibility"],
                "suggested_status": "quarantine",
            }
        )
    elif decision == "FAIL":
        candidates.append(
            {
                "candidate_unit_id": f"auto_{results.get('run_id', 'run')}_failure_signature",
                "reason": "Run failed compile or correctness gates and may be useful as a failure signature.",
                "required_gates": ["schema", "reproducibility"],
                "suggested_status": "quarantine",
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--context")
    parser.add_argument("--trace")
    parser.add_argument("--results", required=True)
    parser.add_argument("--benchmark")
    parser.add_argument("--profile")
    parser.add_argument("--workspace")
    parser.add_argument("--agent-label", default="external_agent")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    task = load_json(args.task, {})
    backend = load_json(args.backend, {})
    context = load_json(args.context, {})
    results = load_json(args.results, {})
    benchmark = load_json(args.benchmark, {})
    profile = load_json(args.profile, {})
    trace_rows = read_trace(args.trace)
    workspace = Path(args.workspace) if args.workspace else Path(args.results).parent

    run_id = results.get("run_id") or workspace.name
    phase_events = trace_rows or results.get("phase_events") or []
    if isinstance(phase_events, dict):
        phase_events = [phase_events]

    record = {
        "run_id": run_id,
        "schema_version": "0.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": {
            "op_family": task.get("op_family") or task.get("operator_family") or task.get("name", "unknown"),
            "op_name": task.get("op_name") or task.get("name"),
            "op_spec_hash": task.get("op_spec_hash") or hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
            "shapes": [str(s) for s in task.get("shapes", task.get("shape", [] if "shape" not in task else [task.get("shape")]))],
            "dtypes": [str(d) for d in task.get("dtypes", task.get("dtype", [] if "dtype" not in task else [task.get("dtype")]))],
            "goal": task.get("goal", "correct_then_fast"),
        },
        "backend_fingerprint": {
            "vendor": backend.get("vendor", "nvidia"),
            "backend": backend.get("backend", "triton_cuda"),
            "device_name": backend.get("device_name") or (backend.get("torch") or {}).get("device_name") or "unknown",
            "device_class": backend.get("device_class", "unknown"),
            "toolchain": backend.get("toolchain", {}),
        },
        "agent_label": args.agent_label,
        "kb_version": args.kb_version,
        "context_packet_hashes": [hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()] if context else [],
        "used_capsules": context.get("capsule_ids", []) if isinstance(context, dict) else [],
        "phase_events": phase_events,
        "final_decision": final_decision(results),
        "decision_rationale": results.get("decision_rationale", "Derived from compile, hidden correctness, and benchmark summary."),
        "cost": results.get("cost", {}),
        "artifacts": {
            "workspace_uri": str(workspace),
            "candidate_uris": [str(workspace / "candidate.py")] if (workspace / "candidate.py").exists() else [],
            "logs": [str(p) for p in [workspace / "compile.log", workspace / "correctness.log"] if p.exists()],
            "benchmark_json": args.benchmark or str(workspace / "benchmark.json"),
            "profile_json": args.profile or str(workspace / "profile_summary.json"),
        },
        "promotion_candidates": promotion_candidates({**results, "run_id": run_id}),
        "artifact_hashes": {"candidate.py": sha256_file(workspace / "candidate.py")},
        "benchmark": benchmark,
        "profile_summary": profile,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
