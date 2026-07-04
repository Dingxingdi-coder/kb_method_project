#!/usr/bin/env python3
"""Build a phase-aware ContextPacket from ECC-KB capsules or baseline modes.

This first implementation uses JSON files and deterministic ranking. It avoids vector DBs on
purpose: the H20 MVP should validate the capsule protocol before optimizing the index layer.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def load_capsules(kb_root: Path, kb_version: str) -> list[dict[str, Any]]:
    roots = [kb_root / "stable" / kb_version, kb_root / "stable", kb_root / "quarantine", kb_root / "failures"]
    capsules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in glob.glob(str(root / "**" / "*.json"), recursive=True):
            try:
                doc = load_json(path)
            except Exception:
                continue
            if not isinstance(doc, dict) or "id" not in doc:
                continue
            cid = str(doc["id"])
            if cid in seen:
                continue
            doc["_source_path"] = path
            capsules.append(doc)
            seen.add(cid)
    return capsules


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def text_contains_any(text: str, keys: list[str]) -> bool:
    hay = text.lower()
    return any(str(k).lower() in hay for k in keys)


def capsule_matches(c: dict[str, Any], task: dict[str, Any], backend: dict[str, Any], phase: str) -> bool:
    phases = set(as_list(c.get("task_phases")))
    if phases and phase not in phases:
        return False

    op_family = str(task.get("op_family") or task.get("operator_family") or "").lower()
    scope = c.get("operator_scope", {}) or {}
    families = [str(x).lower() for x in as_list(scope.get("families"))]
    if families and "all" not in families and op_family and op_family not in families:
        return False

    status = str(c.get("status", ""))
    if status in {"rejected", "stale"} and c.get("unit_type") != "failure_signature":
        return False

    target_text = json.dumps(backend, ensure_ascii=False).lower()
    instances = as_list(c.get("backend_instances"))
    if instances:
        ok = False
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            inst_text = json.dumps(inst, ensure_ascii=False).lower()
            if "unknown" in inst_text or "requires_revalidation" in inst_text:
                ok = True
            if str(inst.get("backend", "")).lower() in target_text:
                ok = True
            if str(inst.get("vendor", "")).lower() in target_text:
                ok = True
        if not ok and c.get("abstraction_level") not in {"portable", "backend_abstract"}:
            return False

    return True


def capsule_score(c: dict[str, Any], task: dict[str, Any], phase: str) -> tuple[int, str]:
    score = int(c.get("retrieval", {}).get("priority", 50))
    evidence_level = c.get("evidence", {}).get("evidence_level", "claim")
    score += {
        "cross_backend_supported": 25,
        "cross_shape_supported": 20,
        "ablation_supported": 18,
        "multi_run": 12,
        "single_run": 6,
        "claim": 0,
    }.get(evidence_level, 0)
    if c.get("status") == "stable":
        score += 15
    if c.get("status") == "candidate":
        score += 5
    if phase in as_list(c.get("task_phases")):
        score += 5
    return (-score, str(c.get("id", "")))


def build_packet(capsules: list[dict[str, Any]], phase: str, topk: int) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "phase": phase,
        "must_obey": [],
        "recommended_actions": [],
        "anti_actions": [],
        "validation_plan": [],
        "stop_conditions": [],
        "evidence_refs": [],
        "capsule_ids": [],
    }
    for c in capsules[:topk]:
        cid = c.get("id")
        action = c.get("action", {}) or {}
        item = {
            "capsule_id": cid,
            "unit_type": c.get("unit_type"),
            "status": c.get("status"),
            "instruction": action.get("instructions", []),
            "expected_effect": c.get("expected_effect", {}),
            "valid_when": (c.get("conditions", {}) or {}).get("valid_when", []),
            "invalid_when": (c.get("conditions", {}) or {}).get("invalid_when", []),
        }
        kind = action.get("kind")
        unit_type = c.get("unit_type")
        if kind == "must_obey" or unit_type in {"op_contract", "legality_constraint"}:
            packet["must_obey"].append(item)
        elif kind == "anti_action" or unit_type == "failure_signature":
            packet["anti_actions"].append(item)
        elif kind == "stop" or unit_type == "stop_capsule":
            packet["stop_conditions"].append(item)
        elif kind == "validate" or unit_type == "validation_policy":
            packet["validation_plan"].append(item)
        else:
            packet["recommended_actions"].append(item)
        packet["capsule_ids"].append(cid)
        packet["evidence_refs"].extend(as_list((c.get("evidence", {}) or {}).get("source_refs")))
    return packet


def baseline_packet(group: str, phase: str) -> dict[str, Any]:
    base = {"phase": phase, "must_obey": [], "recommended_actions": [], "anti_actions": [], "validation_plan": [], "stop_conditions": [], "evidence_refs": [], "capsule_ids": []}
    if group == "A0_no_kb":
        return base
    if group == "A1_plain_rag":
        base["recommended_actions"].append({"source": "plain_rag_placeholder", "instruction": ["Use retrieved documentation chunks only. No evidence-aware gating is applied in this baseline."]})
    if group == "A2_rulebook":
        base["recommended_actions"].extend([
            {"source": "rulebook", "instruction": ["Use masks for non-power-of-two tails."]},
            {"source": "rulebook", "instruction": ["Prefer coalesced contiguous memory access."]},
            {"source": "rulebook", "instruction": ["Use fp32 accumulation for fp16/bf16 reductions when OpSpec requires it."]},
            {"source": "rulebook", "instruction": ["Tune tile size, num_warps, and num_stages after correctness passes."]},
        ])
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--group", default="A3_ecc_kb")
    parser.add_argument("--phase", default="generate")
    parser.add_argument("--kb-root", default="kb")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    task = load_json(args.task, {})
    backend = load_json(args.backend, {})
    if args.group != "A3_ecc_kb":
        packet = baseline_packet(args.group, args.phase)
    else:
        capsules = [c for c in load_capsules(Path(args.kb_root), args.kb_version) if capsule_matches(c, task, backend, args.phase)]
        capsules = sorted(capsules, key=lambda c: capsule_score(c, task, args.phase))
        packet = build_packet(capsules, args.phase, args.topk)
    packet["task_ref"] = args.task
    packet["backend_ref"] = args.backend
    packet["group"] = args.group
    packet["kb_version"] = args.kb_version

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
