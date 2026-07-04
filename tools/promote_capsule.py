#!/usr/bin/env python3
"""Promote quarantined evolution records into draft ECC-KB capsules.

This MVP promotion gate is conservative. It emits candidate JSON capsules into the requested
output directory and leaves human review/schema validation as the next gate.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decision(record: dict[str, Any]) -> str:
    return str(record.get("final_decision", "NEED_MORE_EVIDENCE"))


def has_hidden_pass(record: dict[str, Any]) -> bool:
    bench = record.get("benchmark", {})
    for event in record.get("phase_events", []):
        correctness = event.get("correctness", {}) if isinstance(event, dict) else {}
        if isinstance(correctness, dict) and correctness.get("hidden") == "pass":
            return True
    return bool(bench.get("hidden_correctness_pass") or record.get("hidden_correctness_pass"))


def make_capsule(record: dict[str, Any]) -> dict[str, Any]:
    task = record.get("task", {})
    backend = record.get("backend_fingerprint", {})
    unit_type = "optimization_motif" if decision(record) == "KEEP" else "failure_signature"
    action_kind = "recommend" if unit_type == "optimization_motif" else "anti_action"
    status = "candidate" if decision(record) == "KEEP" and has_hidden_pass(record) else "quarantine"
    cid = f"{unit_type}_{task.get('op_family', 'op')}_{record.get('run_id', 'run')}".lower().replace("/", "_").replace(" ", "_")

    if unit_type == "optimization_motif":
        instructions = [
            "Reuse the successful candidate trajectory only under matching op family, dtype/layout, and backend fingerprint conditions.",
            "Run hidden correctness before benchmarking and require p95 non-regression before promotion to stable.",
        ]
        expected = {"correctness": "Must remain hidden-correct.", "performance": "Expected non-regression or speedup on matching shape bucket.", "search_efficiency": "Warm-starts future candidates from measured trajectory.", "risk": "May overfit to the measured shape/toolchain if valid_when is too broad."}
    else:
        instructions = [
            "Do not repeat the failed action pattern under matching task and backend conditions.",
            "Prefer a minimal semantic or compile repair before attempting performance tuning.",
        ]
        expected = {"correctness": "Avoids repeated compile/correctness failure.", "performance": "No direct performance effect.", "search_efficiency": "Reduces repeated invalid attempts.", "risk": "May reject a valid strategy if the failure signature is too broad."}

    return {
        "id": cid[:120],
        "schema_version": "0.1",
        "unit_type": unit_type,
        "status": status,
        "abstraction_level": "backend_instance",
        "task_phases": ["performance_optimize", "retrospective"] if unit_type == "optimization_motif" else ["compile_repair", "correctness_repair", "retrospective"],
        "operator_scope": {
            "families": [str(task.get("op_family", "unknown"))],
            "shape_patterns": task.get("shapes", []),
            "dtype_scope": task.get("dtypes", []),
        },
        "backend_instances": [
            {
                "vendor": backend.get("vendor", "nvidia"),
                "backend": backend.get("backend", "triton_cuda"),
                "device_class": backend.get("device_class", "unknown"),
                "target_fingerprint": json.dumps(backend, sort_keys=True, ensure_ascii=False),
                "toolchain": backend.get("toolchain", {}),
                "validity": "requires_revalidation",
            }
        ],
        "conditions": {
            "valid_when": ["Same op family, compatible shape bucket, dtype scope, and backend/toolchain fingerprint."],
            "invalid_when": ["Different backend instance, changed toolchain, or hidden correctness fails."],
        },
        "mechanism": record.get("decision_rationale", "Derived from an H20 MVP run."),
        "action": {"kind": action_kind, "instructions": instructions},
        "expected_effect": expected,
        "evidence": {
            "source_refs": [record.get("run_id", "unknown_run")],
            "run_ids": [record.get("run_id", "unknown_run")],
            "benchmark_refs": [record.get("artifacts", {}).get("benchmark_json", "")],
            "profile_refs": [record.get("artifacts", {}).get("profile_json", "")],
            "evidence_level": "single_run",
            "confidence": 0.35,
        },
        "validation": {
            "gates": ["schema", "compile", "hidden_correctness", "benchmark_repeat", "reproducibility"] if unit_type == "optimization_motif" else ["schema", "reproducibility"],
            "correctness_matrix": ["smoke", "quick", "hidden"],
            "promotion_policy": "Human review plus repeated benchmark is required before status=stable.",
        },
        "failure_boundary": {
            "known_failures": [] if unit_type == "optimization_motif" else [record.get("decision_rationale", "Run failed.")],
            "boundary_tests": ["held-out shape sweep", "dtype matrix"],
            "do_not_apply": ["Do not transfer to a new backend without revalidation."],
        },
        "migration": {
            "transfer_policy": "requires_revalidation",
            "portable_part": ["op semantic condition", "failure or optimization mechanism"],
            "replace_on_new_backend": ["tile sizes", "num_warps", "compiler/toolchain assumptions"],
        },
        "retrieval": {"keys": [str(task.get("op_family", "unknown")), str(backend.get("backend", "triton_cuda")), unit_type], "priority": 60 if status == "candidate" else 30, "max_context_tokens": 400},
        "lineage": {"created_by": "tools/promote_capsule.py", "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "parent_run_id": record.get("run_id")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--schema", help="Optional schema path; validate generated capsules if jsonschema is installed.")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in glob.glob(str(Path(args.quarantine) / "**" / "*.json"), recursive=True):
        try:
            doc = load_json(path)
            if isinstance(doc, dict) and "run_id" in doc:
                records.append(doc)
        except Exception:
            continue

    generated = []
    for record in records:
        if decision(record) not in {"KEEP", "FAIL"}:
            continue
        capsule = make_capsule(record)
        path = out_dir / f"{capsule['id']}.json"
        path.write_text(json.dumps(capsule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        generated.append(path)

    if args.schema and generated:
        try:
            import jsonschema  # type: ignore

            schema = load_json(args.schema)
            validator = jsonschema.Draft202012Validator(schema)
            for path in generated:
                errors = list(validator.iter_errors(load_json(path)))
                if errors:
                    print(f"SCHEMA_WARN {path}: {errors[0].message}")
        except Exception as exc:
            print(f"Schema validation skipped: {exc}")

    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# Promotion report\n\n" + "\n".join(f"- {p}" for p in generated) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} capsule(s) in {out_dir}")


if __name__ == "__main__":
    main()
