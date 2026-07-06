#!/usr/bin/env python3
"""Compress one H20 harness run into an ECC-KB evolution record."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ecc_utils import backend_required_fields, read_json, read_jsonl, short_hash, stable_hash, utc_now, write_json


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def hidden_passed(results: dict[str, Any]) -> bool:
    hidden = get_nested(results, "correctness.hidden.status")
    if hidden is None:
        hidden = get_nested(results, "correctness.hidden")
    return str(hidden).lower() in ("pass", "passed", "true", "1")


def compile_passed(results: dict[str, Any]) -> bool:
    status = get_nested(results, "compile.status", get_nested(results, "compile", "unknown"))
    return str(status).lower() in ("pass", "passed", "success", "true", "1")


def best_speedup(results: dict[str, Any], benchmark: dict[str, Any]) -> float:
    candidates = [
        get_nested(results, "benchmark.speedup_vs_eager_p50"),
        get_nested(results, "benchmark.speedup_vs_torch_compile_p50"),
        get_nested(benchmark, "candidate.speedup_vs_eager_p50"),
        get_nested(benchmark, "candidate.speedup_vs_torch_compile_p50"),
        get_nested(benchmark, "speedup_vs_eager_p50"),
        get_nested(benchmark, "speedup_vs_torch_compile_p50"),
    ]
    values = []
    for item in candidates:
        try:
            if item is not None:
                values.append(float(item))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def final_decision(results: dict[str, Any], benchmark: dict[str, Any]) -> str:
    explicit = results.get("final_decision")
    if explicit in {"KEEP", "DISCARD", "FAIL", "NEED_MORE_EVIDENCE"}:
        return explicit
    if not compile_passed(results):
        return "FAIL"
    if not hidden_passed(results):
        return "FAIL"
    if best_speedup(results, benchmark) >= 1.0:
        return "KEEP"
    return "DISCARD"


def event_result_from_status(event: dict[str, Any]) -> str:
    raw = str(event.get("result") or event.get("status") or "").lower()
    if raw in {"compile_fail", "correctness_fail", "performance_regression", "keep_candidate", "discard_candidate", "timeout", "needs_more_evidence"}:
        return raw
    if "compile" in raw and "fail" in raw:
        return "compile_fail"
    if "correct" in raw and "fail" in raw:
        return "correctness_fail"
    if "timeout" in raw:
        return "timeout"
    if "keep" in raw or "pass" in raw:
        return "keep_candidate"
    if "regression" in raw:
        return "performance_regression"
    return "discard_candidate"


def build_phase_events(trace: list[dict[str, Any]], results: dict[str, Any], benchmark: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if trace:
        for idx, event in enumerate(trace):
            events.append(
                {
                    "iteration": int(event.get("iteration", idx)),
                    "phase": str(event.get("phase", "unknown")),
                    "action_summary": str(event.get("action_summary") or event.get("action") or "not recorded"),
                    "candidate_hash": str(event.get("candidate_hash", "")),
                    "compile": event.get("compile", {}),
                    "correctness": event.get("correctness", {}),
                    "benchmark": event.get("benchmark", {}),
                    "profile_summary": event.get("profile_summary", {}),
                    "result": event_result_from_status(event),
                    "diagnosis": str(event.get("diagnosis", "")),
                }
            )
    else:
        result = "keep_candidate" if final_decision(results, benchmark) == "KEEP" else "discard_candidate"
        if not compile_passed(results):
            result = "compile_fail"
        elif not hidden_passed(results):
            result = "correctness_fail"
        events.append(
            {
                "iteration": int(get_nested(results, "cost.iterations", 0) or 0),
                "phase": str(results.get("phase", "final")),
                "action_summary": "harness result ingestion",
                "candidate_hash": str(results.get("candidate_hash", "")),
                "compile": results.get("compile", {}),
                "correctness": results.get("correctness", {}),
                "benchmark": benchmark,
                "profile_summary": profile,
                "result": result,
                "diagnosis": str(results.get("diagnosis", "")),
            }
        )
    return events


def build_promotion_candidates(task: dict[str, Any], results: dict[str, Any], benchmark: dict[str, Any], decision: str) -> list[dict[str, Any]]:
    op = str(task.get("op_family", "op"))
    task_hash = str(task.get("op_spec_hash") or short_hash(task))
    candidates: list[dict[str, Any]] = []
    if decision == "KEEP" and hidden_passed(results):
        candidates.append(
            {
                "candidate_unit_id": f"auto_perf_{op}_{task_hash}_{short_hash(benchmark or results)}",
                "reason": "Candidate passed hidden correctness and produced a benchmark result eligible for promotion review.",
                "required_gates": ["schema", "compile", "hidden_correctness", "benchmark_repeat", "profile_attribution", "reproducibility"],
                "suggested_status": "candidate",
            }
        )
    elif decision == "FAIL":
        candidates.append(
            {
                "candidate_unit_id": f"auto_fail_{op}_{task_hash}_{short_hash(results)}",
                "reason": "Run failed and may define a reusable failure signature if the same diagnosis repeats.",
                "required_gates": ["schema", "reproducibility"],
                "suggested_status": "quarantine",
            }
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--agent-label", default="unknown_agent")
    parser.add_argument("--kb-version", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    task = read_json(args.task)
    context = read_json(args.context, default={})
    results = read_json(args.results)
    backend = read_json(args.backend)
    trace = read_jsonl(args.trace)
    benchmark = read_json(args.benchmark, default={}) if args.benchmark else {}
    profile = read_json(args.profile, default={}) if args.profile else {}

    run_id = args.run_id or str(results.get("run_id") or f"run_{short_hash([task, context, results])}")
    decision = final_decision(results, benchmark)
    phase_events = build_phase_events(trace, results, benchmark, profile)

    context_hash = str(context.get("context_packet_hash") or stable_hash(context))
    used_capsules = []
    for key in ("retrieved_capsules", "used_capsules"):
        if isinstance(context.get(key), list):
            used_capsules.extend(str(x) for x in context.get(key, []))
    if isinstance(results.get("used_capsules"), list):
        used_capsules.extend(str(x) for x in results.get("used_capsules", []))
    used_capsules = sorted(set(used_capsules))

    cost = results.get("cost", {}) if isinstance(results.get("cost"), dict) else {}
    wall_time_s = float(cost.get("wall_time_s", 0) or results.get("wall_time_s", 0) or 0)
    iterations = int(cost.get("iterations", 0) or results.get("iterations", len(phase_events)) or len(phase_events))
    gpu_benchmark_runs = int(cost.get("gpu_benchmark_runs", 0) or get_nested(benchmark, "measurement.repeats", 0) or 0)

    record = {
        "run_id": run_id,
        "schema_version": "0.1",
        "timestamp": utc_now(),
        "task": {
            "op_family": str(task.get("op_family", "unknown")),
            "op_name": str(task.get("op_name", task.get("variant", ""))),
            "op_spec_hash": str(task.get("op_spec_hash") or short_hash(task)),
            "shapes": [str(task.get("shape", ""))],
            "dtypes": [str(task.get("dtype", ""))],
            "goal": str(task.get("goal", "compile_correct_and_benchmark")),
        },
        "backend_fingerprint": backend_required_fields(backend),
        "agent_label": args.agent_label,
        "kb_version": args.kb_version or str(context.get("kb_version") or backend.get("repo", {}).get("kb_version", "unknown")),
        "context_packet_hashes": [context_hash],
        "used_capsules": used_capsules,
        "phase_events": phase_events,
        "final_decision": decision,
        "decision_rationale": str(results.get("decision_rationale") or results.get("diagnosis") or ""),
        "cost": {
            "wall_time_s": wall_time_s,
            "agent_tokens_in": int(cost.get("agent_tokens_in", 0) or 0),
            "agent_tokens_out": int(cost.get("agent_tokens_out", 0) or 0),
            "gpu_benchmark_runs": gpu_benchmark_runs,
            "iterations": iterations,
        },
        "artifacts": {
            "workspace_uri": str(Path(args.results).parent),
            "candidate_uris": [str(Path(args.results).parent / "candidate.py")],
            "logs": [str(Path(args.trace)), str(Path(args.results))],
            "benchmark_json": str(args.benchmark or ""),
            "profile_json": str(args.profile or ""),
        },
        "promotion_candidates": build_promotion_candidates(task, results, benchmark, decision),
    }

    write_json(args.out, record)
    print(f"wrote {args.out} decision={decision} run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
