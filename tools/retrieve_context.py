#!/usr/bin/env python3
"""Build retrieval context for one H20 experimental group.

A0 returns no knowledge.
A1 returns vector-RAG chunks from the frozen raw corpus.
A2 returns vector-RAG chunks from KB knowledge units.
A3 returns ECC-KB evidence capsules filtered by phase, task, backend, and evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ecc_utils import iter_json_files, normalize_shape, read_json, safe_id, shape_bucket, short_hash, stable_hash, write_json


GROUP_ALIASES = {
    "A0_no_kb": "A0_prompt",
    "A1_plain_rag": "A1_raw_corpus_vector_rag",
    "A1_raw_corpus_rag": "A1_raw_corpus_vector_rag",
    "A2_rulebook": "A2_kb_vector_rag",
    "A2_kb_plain_rag": "A2_kb_vector_rag",
}

GROUPS = ["A0_prompt", "A1_raw_corpus_vector_rag", "A2_kb_vector_rag", "A3_ecc_kb", *GROUP_ALIASES]

RAW_CORPUS_VERSION = "raw_corpus_v0"
RAW_CORPUS_VECTOR_INDEX_VERSION = "raw_corpus_vector_index_v0"
KB_VECTOR_INDEX_VERSION = "kb_vector_index_v0"
ECC_KB_INDEX_VERSION = "ecc_kb_index_v0_structured"


def canonical_group(group: str) -> str:
    return GROUP_ALIASES.get(group, group)


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


def tokenize(text: str) -> list[str]:
    return [safe_id(t) for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if t]


def query_terms(task: dict[str, Any], phase: str) -> set[str]:
    parts = [
        phase,
        str(task.get("task_id", "")),
        str(task.get("op_family", "")),
        str(task.get("op_name", "")),
        str(task.get("variant", "")),
        str(task.get("dtype", "")),
        str(task.get("goal", "")),
        normalize_shape(task.get("shape") or task.get("shapes")),
        shape_bucket(task.get("shape") or task.get("shapes")),
        json.dumps(task.get("op_spec", {}), ensure_ascii=False, sort_keys=True),
    ]
    terms = set(tokenize(" ".join(parts)))
    op_family = str(task.get("op_family", "")).lower()
    if op_family == "softmax":
        terms.update({"softmax", "exp", "max", "mask", "padded", "denominator", "stable"})
    elif op_family == "reduction":
        terms.update({"reduction", "reduce", str(task.get("op_name", "")).lower(), "mask", "sum", "max"})
    elif op_family == "layernorm":
        terms.update({"layernorm", "mean", "variance", "eps", "gamma", "beta"})
    elif op_family == "matmul":
        terms.update({"matmul", "gemm", "dot", "tile", "block"})
    return {safe_id(t) for t in terms if t}


def score_text(text: str, terms: set[str]) -> tuple[int, int]:
    words = tokenize(text)
    if not words:
        return (0, 0)
    counts = {word: words.count(word) for word in set(words)}
    overlap = sum(1 for term in terms if term in counts)
    frequency = sum(min(counts.get(term, 0), 4) for term in terms)
    return (overlap, frequency)


def query_text(task: dict[str, Any], phase: str) -> str:
    parts = [
        f"phase: {phase}",
        f"task_id: {task.get('task_id', '')}",
        f"operator: {task.get('op_family', '')} {task.get('op_name', '')} {task.get('variant', '')}",
        f"dtype: {task.get('dtype', '')}",
        f"shape: {normalize_shape(task.get('shape') or task.get('shapes'))}",
        f"shape_bucket: {shape_bucket(task.get('shape') or task.get('shapes'))}",
        f"goal: {task.get('goal', '')}",
        f"op_spec: {json.dumps(task.get('op_spec', {}), ensure_ascii=False, sort_keys=True)}",
    ]
    op_family = str(task.get("op_family", "")).lower()
    if op_family == "softmax":
        parts.append("softmax stable exp row max mask padded denominator fp32 accumulation triton")
    elif op_family == "reduction":
        parts.append("row reduction sum max mask neutral value fp32 accumulation triton")
    elif op_family == "layernorm":
        parts.append("layernorm mean variance eps affine fp32 accumulation triton")
    elif op_family == "matmul":
        parts.append("matmul gemm dot tile boundary mask num warps num stages triton")
    return "\n".join(str(p) for p in parts if p)


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def split_markdown_chunks(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        chunks.append(current)
    return chunks


def raw_corpus_paths(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for subdir in ("raw_corpus", "raw_archive"):
        base = source_root / subdir
        if base.exists():
            paths.extend(sorted(base.glob("*.md")))
    return paths


def load_vector_index(index_dir: Path) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    manifest_path = index_dir / "manifest.json"
    npz_path = index_dir / "index.npz"
    if not manifest_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"missing vector index at {index_dir}; build it with tools/build_vector_index.py before running vector-RAG groups"
        )
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency checked in integration use
        raise RuntimeError("vector retrieval requires numpy") from exc
    manifest = read_json(manifest_path)
    data = np.load(npz_path, allow_pickle=False)
    embeddings = data["embeddings"].astype("float32", copy=False)
    items = manifest.get("items", [])
    if len(items) != int(embeddings.shape[0]):
        raise ValueError(f"{index_dir}: manifest item count does not match embeddings")
    return manifest, embeddings, items


def embed_query(text: str, model_path: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency checked in integration use
        raise RuntimeError("vector retrieval requires sentence-transformers") from exc
    model = SentenceTransformer(model_path, device="cpu")
    return model.encode([text], normalize_embeddings=True)[0].astype("float32", copy=False)


def vector_rank(index_dir: Path, task: dict[str, Any], phase: str, topk: int, embedding_model_path: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, embeddings, items = load_vector_index(index_dir)
    model_path = embedding_model_path or manifest.get("embedding_model_path")
    if not model_path:
        raise ValueError(f"{index_dir}: manifest is missing embedding_model_path; pass --embedding-model-path")
    query = query_text(task, phase)
    query_embedding = embed_query(query, str(model_path))
    scores = embeddings @ query_embedding
    ranked = sorted(range(len(items)), key=lambda idx: float(scores[idx]), reverse=True)[:topk]
    retrieved: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked, start=1):
        item = dict(items[idx])
        item["score"] = {"cosine": float(scores[idx]), "rank": rank}
        retrieved.append(item)
    return manifest, retrieved


def vector_packet_common(packet: dict[str, Any], manifest: dict[str, Any], index_dir: Path) -> None:
    packet["embedding_model_id"] = manifest.get("embedding_model_id", "")
    packet["embedding_model_path"] = manifest.get("embedding_model_path", "")
    packet["vector_index_version"] = manifest.get("index_version", "")
    packet["vector_index_hash"] = manifest.get("index_hash", "")
    packet["vector_index_path"] = str(index_dir)
    packet["vector_index_kind"] = manifest.get("kind", "")
    packet["retrieval_topk"] = packet.get("retrieval_topk")


def build_vector_raw_rag_packet(task: dict[str, Any], phase: str, index_dir: Path, topk: int, embedding_model_path: str | None) -> dict[str, Any]:
    manifest, chunks = vector_rank(index_dir, task, phase, topk, embedding_model_path)
    packet = {
        "schema_version": "0.1",
        "context_packet_hash": "",
        "group": "A1_raw_corpus_vector_rag",
        "phase": phase,
        "task_id": task.get("task_id"),
        "retrieval_mode": "vector_rag",
        "source_corpus_version": RAW_CORPUS_VERSION,
        "raw_corpus_index_version": manifest.get("index_version", RAW_CORPUS_VECTOR_INDEX_VERSION),
        "retrieval_topk": topk,
        "retrieved_items": chunks,
        "retrieved_item_ids": [c["id"] for c in chunks],
        "document_chunks": chunks,
        "must_obey": [],
        "recommended_actions": [{"capsule_id": c["id"], "instruction": c["text"], "source_path": c["source_path"]} for c in chunks],
        "anti_actions": [],
        "validation_plan": [],
        "stop_conditions": [],
        "evidence_refs": [{"id": c["id"], "source_path": c["source_path"], "source_type": c["source_type"]} for c in chunks],
    }
    vector_packet_common(packet, manifest, index_dir)
    return packet


def build_plain_raw_rag_packet(task: dict[str, Any], phase: str, root: Path, source_root: Path, topk: int, max_chars: int) -> dict[str, Any]:
    terms = query_terms(task, phase)
    scored: list[tuple[tuple[int, int], int, dict[str, Any]]] = []
    for path in raw_corpus_paths(source_root):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        source_type = "raw_archive" if path.parent.name == "raw_archive" else "raw_corpus"
        for idx, chunk in enumerate(split_markdown_chunks(text, max_chars=max_chars)):
            score = score_text(f"{path.name}\n{chunk}", terms)
            if score == (0, 0):
                continue
            rel = relative_path(path, root)
            item = {
                "id": f"raw:{safe_id(path.stem)}:{idx}",
                "source_type": source_type,
                "source_path": rel,
                "text": chunk[:max_chars],
                "score": {"term_overlap": score[0], "term_frequency": score[1]},
            }
            scored.append((score, -idx, item))
    scored.sort(key=lambda row: row[:2], reverse=True)
    chunks = [item for _, _, item in scored[:topk]]
    return {
        "schema_version": "0.1",
        "context_packet_hash": "",
        "group": "A1_raw_corpus_rag",
        "phase": phase,
        "task_id": task.get("task_id"),
        "retrieval_mode": "plain_text_rag",
        "source_corpus_version": RAW_CORPUS_VERSION,
        "raw_corpus_index_version": "raw_corpus_rag_index_v0_simple_text",
        "retrieved_items": chunks,
        "retrieved_item_ids": [c["id"] for c in chunks],
        "document_chunks": chunks,
        "must_obey": [],
        "recommended_actions": [{"capsule_id": c["id"], "instruction": c["text"], "source_path": c["source_path"]} for c in chunks],
        "anti_actions": [],
        "validation_plan": [],
        "stop_conditions": [],
        "evidence_refs": [{"id": c["id"], "source_path": c["source_path"], "source_type": c["source_type"]} for c in chunks],
    }


def kb_unit_to_plain_text(unit: dict[str, Any]) -> str:
    action = unit.get("action", {}) if isinstance(unit.get("action"), dict) else {}
    evidence = unit.get("evidence", {}) if isinstance(unit.get("evidence"), dict) else {}
    validation = unit.get("validation", {}) if isinstance(unit.get("validation"), dict) else {}
    boundary = unit.get("failure_boundary", {}) if isinstance(unit.get("failure_boundary"), dict) else {}
    conditions = unit.get("conditions", {}) if isinstance(unit.get("conditions"), dict) else {}
    scope = unit.get("operator_scope", {}) if isinstance(unit.get("operator_scope"), dict) else {}
    lines = [
        f"id: {unit.get('id')}",
        f"unit_type: {unit.get('unit_type')}",
        f"status: {unit.get('status')}",
        f"abstraction_level: {unit.get('abstraction_level')}",
        f"task_phases: {', '.join(str(x) for x in unit.get('task_phases', []))}",
        f"operator_scope: {json.dumps(scope, ensure_ascii=False, sort_keys=True)}",
        f"valid_when: {'; '.join(str(x) for x in conditions.get('valid_when', []))}",
        f"invalid_when: {'; '.join(str(x) for x in conditions.get('invalid_when', []))}",
        f"action_kind: {action.get('kind')}",
        f"instructions: {'; '.join(str(x) for x in action.get('instructions', []))}",
        f"expected_effect: {json.dumps(unit.get('expected_effect', {}), ensure_ascii=False, sort_keys=True)}",
        f"evidence_level: {evidence.get('evidence_level')}",
        f"source_refs: {', '.join(str(x) for x in evidence.get('source_refs', []))}",
        f"validation_gates: {', '.join(str(x) for x in validation.get('gates', []))}",
        f"failure_boundary: {json.dumps(boundary, ensure_ascii=False, sort_keys=True)}",
    ]
    if unit.get("mechanism"):
        lines.append(f"mechanism: {unit.get('mechanism')}")
    return "\n".join(lines)


def build_vector_kb_rag_packet(task: dict[str, Any], phase: str, index_dir: Path, topk: int, embedding_model_path: str | None) -> dict[str, Any]:
    manifest, units = vector_rank(index_dir, task, phase, topk, embedding_model_path)
    packet = {
        "schema_version": "0.1",
        "context_packet_hash": "",
        "group": "A2_kb_vector_rag",
        "phase": phase,
        "task_id": task.get("task_id"),
        "retrieval_mode": "vector_rag",
        "source_corpus_version": RAW_CORPUS_VERSION,
        "kb_version": task.get("kb_version"),
        "kb_vector_index_version": manifest.get("index_version", KB_VECTOR_INDEX_VERSION),
        "retrieval_topk": topk,
        "retrieved_items": units,
        "retrieved_item_ids": [u["id"] for u in units],
        "kb_units": units,
        "must_obey": [],
        "recommended_actions": [{"capsule_id": u["id"], "instruction": u["text"], "source_path": u["source_path"]} for u in units],
        "anti_actions": [],
        "validation_plan": [],
        "stop_conditions": [],
        "evidence_refs": [{"id": u["id"], "source_path": u["source_path"], "source_type": u["source_type"]} for u in units],
    }
    vector_packet_common(packet, manifest, index_dir)
    return packet


def build_plain_kb_rag_packet(task: dict[str, Any], phase: str, root: Path, kb_paths: list[Path], topk: int, max_chars: int) -> dict[str, Any]:
    terms = query_terms(task, phase)
    scored: list[tuple[tuple[int, int], int, dict[str, Any]]] = []
    for idx, unit in enumerate(load_capsules(kb_paths)):
        if unit.get("status") in ("rejected", "stale"):
            continue
        text = kb_unit_to_plain_text(unit)
        score = score_text(text, terms)
        if score == (0, 0):
            continue
        unit_id = str(unit.get("id") or f"kb_unit_{idx}")
        source_path = relative_path(Path(str(unit.get("_source_path", ""))), root)
        item = {
            "id": unit_id,
            "source_type": "kb_unit",
            "source_path": source_path,
            "text": text[:max_chars],
            "score": {"term_overlap": score[0], "term_frequency": score[1]},
        }
        priority = int(unit.get("retrieval", {}).get("priority", 0) or 0)
        scored.append((score, priority, item))
    scored.sort(key=lambda row: row[:2], reverse=True)
    units = [item for _, _, item in scored[:topk]]
    return {
        "schema_version": "0.1",
        "context_packet_hash": "",
        "group": "A2_kb_plain_rag",
        "phase": phase,
        "task_id": task.get("task_id"),
        "retrieval_mode": "plain_text_rag",
        "source_corpus_version": RAW_CORPUS_VERSION,
        "kb_version": task.get("kb_version"),
        "kb_plain_rag_index_version": "kb_plain_rag_index_v0_simple_text",
        "retrieved_items": units,
        "retrieved_item_ids": [u["id"] for u in units],
        "kb_units": units,
        "must_obey": [],
        "recommended_actions": [{"capsule_id": u["id"], "instruction": u["text"], "source_path": u["source_path"]} for u in units],
        "anti_actions": [],
        "validation_plan": [],
        "stop_conditions": [],
        "evidence_refs": [{"id": u["id"], "source_path": u["source_path"], "source_type": u["source_type"]} for u in units],
    }


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
        "retrieval_mode": "phase_and_evidence_aware_context_packet",
        "source_corpus_version": RAW_CORPUS_VERSION,
        "kb_version": task.get("kb_version"),
        "ecc_kb_index_version": ECC_KB_INDEX_VERSION,
        "retrieved_items": [],
        "retrieved_item_ids": [],
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
        packet["retrieved_item_ids"].append(unit.get("id"))
        packet["retrieved_items"].append(
            {
                "id": unit.get("id"),
                "source_type": "ecc_capsule",
                "source_path": unit.get("_source_path"),
                "unit_type": unit.get("unit_type"),
                "status": unit.get("status"),
                "action_kind": action.get("kind"),
            }
        )

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
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--phase", default="generate")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--kb-root", default="kb")
    parser.add_argument("--source-root", default="sources")
    parser.add_argument("--raw-vector-index", default="artifacts/indexes/raw_corpus_vector_v0")
    parser.add_argument("--kb-vector-index", default="artifacts/indexes/kb_vector_v0")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--max-chars-per-item", type=int, default=1800)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    task = read_json(args.task)
    backend = read_json(args.backend)
    task["kb_version"] = args.kb_version
    group = canonical_group(args.group)

    if group == "A0_prompt":
        packet = {
            "schema_version": "0.1",
            "context_packet_hash": "",
            "group": group,
            "phase": args.phase,
            "task_id": task.get("task_id"),
            "retrieval_mode": "none",
            "source_corpus_version": "none",
            "raw_corpus_index_version": "none",
            "kb_version": "none",
            "kb_plain_rag_index_version": "none",
            "kb_vector_index_version": "none",
            "ecc_kb_index_version": "none",
            "vector_index_version": "none",
            "vector_index_hash": "none",
            "retrieved_items": [],
            "retrieved_item_ids": [],
            "must_obey": [],
            "recommended_actions": [],
            "anti_actions": [],
            "validation_plan": [],
            "stop_conditions": [],
            "evidence_refs": [],
        }
    elif group == "A1_raw_corpus_vector_rag":
        packet = build_vector_raw_rag_packet(task, args.phase, Path(args.raw_vector_index), args.topk, args.embedding_model_path)
    elif group == "A2_kb_vector_rag":
        packet = build_vector_kb_rag_packet(task, args.phase, Path(args.kb_vector_index), args.topk, args.embedding_model_path)
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
    print(f"wrote {args.out} group={group} phase={args.phase} hash={packet['context_packet_short_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
