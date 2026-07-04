#!/usr/bin/env python3
"""Build a ContextPacket for one experimental group.

A0 returns no knowledge.
A1 returns plain document snippets.
A2 returns a static rulebook.
A3 returns ECC-KB evidence capsules filtered by phase, task, backend, and evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ecc_utils import iter_json_files, normalize_shape, read_json, safe_id, shape_bucket, short_hash, stable_hash, write_json


DEFAULT_RULEBOOK = [
    {
        "capsule_id": "rule_tail_mask",
        "instruction": "Use explicit masks for non-power-of-two tails and boundary elements.",
        "kind": "must_obey",
    },
    {
        "capsule_id": "rule_fp32_accumulation",
        "instruction": "Use fp32 accumulation for fp16/bf16 reductions unless the OpSpec declares another policy.",
        "kind": "must_obey",
    },
    {
        "capsule_id": "rule_no_hardcode_visible_shapes",
        "instruction": "Do not hardcode public shapes or constants derived only from visible tests.",
        "kind": "anti_action",
    },
    {
        "capsule_id": "rule_tune_tile_after_correctness",
        "instruction": "Tune tile size, num_warps, and stages only after hidden correctness passes.",
        "kind": "recommend",
    },
    {
        "capsule_id": "rule_report_p50_p95",
        "instruction": "Report p50 and p95 latency against eager and torch.compile baselines.",
        "kind": "validate",
    },
]


def task_terms(task: dict[str, Any], phase: str) -> set[str]:
    op_family = str(task.get("op_family", ""))
    op_name = str(task.get("op_name", ""))
    dtype = str(task.get("dtype", ""))
    shape = task.get("shape") or task.get("shapes")
    terms = {phase, op_family, op_name, dtype, normalize_shape(shape), shape_bucket(shape)}
    if task.get("variant"):
        terms.add(str(task["variant"]))
    if task.get("goal"):
        terms.update(str(task["goal"]).split())
    return {safe_id(t) for t in terms if t}


def doc_snippets(task: dict[str, Any], root: Path, max_chars: int = 1800) -> list[dict[str, str]]:
    """A deliberately simple A1 baseline: text snippets without evidence gates."""
    op_family = str(task.get("op_family", "")).lower()
    candidate_files = [
        root / "docs" / "method.md",
        root / "docs" / "h20_mvp_protocol.md",
        root / "experiments" / "h20" / "operators.md",
        root / "experiments" / "h20" / "baselines.md",
    ]
    snippets: list[dict[str, str]] = []
    for path in candidate_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            lower = para.lower()
            if op_family and op_family in lower:
                snippets.append({"source": str(path), "text": para[:max_chars]})
            elif any(k in lower for k in ("triton", "correctness", "profile", "benchmark")):
                snippets.append({"source": str(path), "text": para[:max_chars]})
            if len(snippets) >= 8:
                return snippets
    return snippets[:8]


def capsule_phase_match(unit: dict[str, Any], phase: str) -> bool:
    phases = set(unit.get("task_phases", []))
    return phase in phases or "all" in phases


def capsule_op_match(unit: dict[str, Any], task: dict[str, Any]) -> bool:
    scope = unit.get("operator_scope", {})
    families = {str(x).lower() for x in scope.get("families", [])}
    op_family = str(task.get("op_family", "")).lower()
    if families and "*" not in families and op_family not in families:
        return False

    dtype_scope = {str(x).lower() for x in scope.get("dtype_scope", [])}
    dtype = str(task.get("dtype", "")).lower()
    if dtype_scope and "*" not in dtype_scope and dtype not in dtype_scope:
        return False

    return True


def capsule_backend_match(unit: dict[str, Any], backend: dict[str, Any]) -> bool:
    abstraction = unit.get("abstraction_level")
    if abstraction in ("portable", "backend_abstract"):
        return True

    instances = unit.get("backend_instances") or []
    if not instances:
        return abstraction != "backend_instance"

    backend_name = str(backend.get("backend", "")).lower()
    vendor = str(backend.get("vendor", "")).lower()
    for instance in instances:
        if str(instance.get("validity", "unknown")) == "invalid":
            continue
        if instance.get("vendor") and str(instance["vendor"]).lower() != vendor:
            continue
        if instance.get("backend") and str(instance["backend"]).lower() not in backend_name:
            continue
        return True
    return False


def rank_capsule(unit: dict[str, Any], terms: set[str]) -> tuple[int, int, int]:
    retrieval = unit.get("retrieval", {})
    keys = {safe_id(k) for k in retrieval.get("keys", [])}
    overlap = len(keys & terms)
    evidence_level = str(unit.get("evidence", {}).get("evidence_level", "claim"))
    evidence_weight = {
        "cross_backend_supported": 6,
        "cross_shape_supported": 5,
        "ablation_supported": 4,
        "multi_run": 3,
        "single_run": 2,
        "claim": 1,
    }.get(evidence_level, 0)
    status_weight = {
        "stable": 5,
        "candidate": 3,
        "quarantine": 1,
        "draft": 0,
        "stale": -2,
        "rejected": -5,
    }.get(str(unit.get("status", "")), 0)
    priority = int(retrieval.get("priority", 0))
    return (status_weight + evidence_weight + overlap, priority, overlap)


def load_capsules(paths: list[Path]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for path in iter_json_files(paths):
        try:
            unit = read_json(path)
            if isinstance(unit, dict):
                unit["_source_path"] = str(path)
                units.append(unit)
        except Exception:
            continue
    return units


def build_a3_packet(task: dict[str, Any], backend: dict[str, Any], phase: str, kb_paths: list[Path], topk: int) -> dict[str, Any]:
    terms = task_terms(task, phase)
    units = []
    for unit in load_capsules(kb_paths):
        if unit.get("status") in ("rejected", "stale"):
            continue
        if not capsule_phase_match(unit, phase):
            continue
        if not capsule_op_match(unit, task):
            continue
        if not capsule_backend_match(unit, backend):
            continue
        units.append(unit)

    units.sort(key=lambda u: rank_capsule(u, terms), reverse=True)
    units = units[:topk]

    packet: dict[str, Any] = {
        "schema_version": "0.1",
        "context_packet_hash": "",
        "group": "A3_ecc_kb",
        "phase": phase,
        "task_id": task.get("task_id"),
        "op_family": task.get("op_family"),
        "kb_version": task.get("kb_version"),
        "must_obey": [],
        "recommended_actions": [],
        "anti_actions": [],
        "validation_plan": [],
        "stop_conditions": [],
        "evidence_refs": [],
        "retrieved_capsules": [],
    }

    for unit in units:
        action = unit.get("action", {})
        item = {
            "capsule_id": unit.get("id"),
            "instruction": "; ".join(action.get("instructions", [])),
            "expected_effect": unit.get("expected_effect", {}),
            "source_path": unit.get("_source_path"),
        }
        kind = action.get("kind")
        if kind == "must_obey":
            packet["must_obey"].append(item)
        elif kind in ("recommend", "repair", "tune"):
            packet["recommended_actions"].append(item)
        elif kind == "anti_action":
            packet["anti_actions"].append(item)
        elif kind == "validate":
            packet["validation_plan"].append(item)
        elif kind == "stop":
            packet["stop_conditions"].append(item)
        else:
            packet["recommended_actions"].append(item)

        packet["evidence_refs"].append(
            {
                "capsule_id": unit.get("id"),
                "evidence": unit.get("evidence", {}),
                "validation": unit.get("validation", {}),
                "failure_boundary": unit.get("failure_boundary", {}),
            }
        )
        packet["retrieved_capsules"].append(unit.get("id"))

    if not packet["must_obey"]:
        packet["must_obey"].extend(
            [
                {"capsule_id": "fallback_no_reference", "instruction": "Do not call PyTorch reference or fallback libraries from candidate.py."},
                {"capsule_id": "fallback_hidden_first", "instruction": "Treat hidden correctness as a hard gate before performance claims."},
            ]
        )
    if not packet["validation_plan"]:
        packet["validation_plan"].append(
            {
                "capsule_id": "fallback_validation_plan",
                "instruction": "Run smoke, quick, hidden correctness, then p50/p95 benchmark with fixed warmup/repeats.",
            }
        )

    packet["context_packet_hash"] = stable_hash(packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--group", choices=["A0_no_kb", "A1_plain_rag", "A2_rulebook", "A3_ecc_kb"], required=True)
    parser.add_argument("--phase", default="generate")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--kb-root", default="kb")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    task = read_json(args.task)
    backend = read_json(args.backend)
    task["kb_version"] = args.kb_version

    if args.group == "A0_no_kb":
        packet = {
            "schema_version": "0.1",
            "context_packet_hash": "",
            "group": args.group,
            "phase": args.phase,
            "task_id": task.get("task_id"),
            "must_obey": [],
            "recommended_actions": [],
            "anti_actions": [],
            "validation_plan": [],
            "stop_conditions": [],
            "evidence_refs": [],
        }
    elif args.group == "A1_plain_rag":
        snippets = doc_snippets(task, root)
        packet = {
            "schema_version": "0.1",
            "context_packet_hash": "",
            "group": args.group,
            "phase": args.phase,
            "task_id": task.get("task_id"),
            "document_chunks": snippets,
            "must_obey": [],
            "recommended_actions": [{"capsule_id": f"rag_chunk_{idx}", "instruction": s["text"]} for idx, s in enumerate(snippets)],
            "anti_actions": [],
            "validation_plan": [],
            "stop_conditions": [],
            "evidence_refs": [{"source": s["source"]} for s in snippets],
        }
    elif args.group == "A2_rulebook":
        packet = {
            "schema_version": "0.1",
            "context_packet_hash": "",
            "group": args.group,
            "phase": args.phase,
            "task_id": task.get("task_id"),
            "must_obey": [r for r in DEFAULT_RULEBOOK if r["kind"] == "must_obey"],
            "recommended_actions": [r for r in DEFAULT_RULEBOOK if r["kind"] == "recommend"],
            "anti_actions": [r for r in DEFAULT_RULEBOOK if r["kind"] == "anti_action"],
            "validation_plan": [r for r in DEFAULT_RULEBOOK if r["kind"] == "validate"],
            "stop_conditions": [],
            "evidence_refs": [],
        }
    else:
        kb_root = Path(args.kb_root)
        kb_paths = [
            kb_root / "stable" / args.kb_version,
            kb_root / "stable",
            kb_root / "quarantine",
            kb_root / "failures",
        ]
        packet = build_a3_packet(task, backend, args.phase, kb_paths, args.topk)

    packet["context_packet_hash"] = stable_hash(packet)
    packet["context_packet_short_hash"] = short_hash(packet)
    write_json(args.out, packet)
    print(f"wrote {args.out} group={args.group} phase={args.phase} hash={packet['context_packet_short_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
