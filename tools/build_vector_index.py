#!/usr/bin/env python3
"""Build local embedding indexes for H20 vector-RAG baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ecc_utils import sha256_file, stable_hash, utc_now, write_json
from retrieve_context import (
    KB_VECTOR_INDEX_VERSION,
    RAW_CORPUS_VECTOR_INDEX_VERSION,
    kb_unit_to_plain_text,
    load_capsules,
    raw_corpus_paths,
    relative_path,
    split_markdown_chunks,
)


def raw_items(repo_root: Path, source_root: Path, max_chars: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in raw_corpus_paths(source_root):
        text = path.read_text(encoding="utf-8")
        source_type = "raw_archive" if path.parent.name == "raw_archive" else "raw_corpus"
        for idx, chunk in enumerate(split_markdown_chunks(text, max_chars=max_chars)):
            item_id = f"raw:{path.stem.lower().replace('.', '_')}:{idx}"
            items.append(
                {
                    "id": item_id,
                    "source_type": source_type,
                    "source_path": relative_path(path, repo_root),
                    "chunk_index": idx,
                    "text": chunk[:max_chars],
                    "content_hash": stable_hash({"source_path": str(path), "chunk_index": idx, "text": chunk[:max_chars]}),
                }
            )
    return items


def kb_items(repo_root: Path, kb_root: Path, kb_version: str, max_chars: int) -> list[dict[str, Any]]:
    kb_paths = [
        kb_root / "stable" / kb_version,
        kb_root / "stable",
        kb_root / "quarantine",
        kb_root / "failures",
    ]
    items: list[dict[str, Any]] = []
    for idx, unit in enumerate(load_capsules(kb_paths)):
        if unit.get("status") in ("rejected", "stale"):
            continue
        unit_id = str(unit.get("id") or f"kb_unit_{idx}")
        text = kb_unit_to_plain_text(unit)[:max_chars]
        source_path = relative_path(Path(str(unit.get("_source_path", ""))), repo_root)
        items.append(
            {
                "id": unit_id,
                "source_type": "kb_unit",
                "source_path": source_path,
                "unit_type": unit.get("unit_type"),
                "status": unit.get("status"),
                "text": text,
                "content_hash": stable_hash({"source_path": source_path, "id": unit_id, "text": text}),
            }
        )
    return items


def encode_texts(model_path: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency checked in integration use
        raise RuntimeError("build_vector_index.py requires sentence-transformers") from exc
    model = SentenceTransformer(model_path, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embeddings, dtype=np.float32)


def build(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.kind == "raw_corpus":
        items = raw_items(repo_root, Path(args.source_root).resolve(), args.max_chars_per_item)
        index_version = args.index_version or RAW_CORPUS_VECTOR_INDEX_VERSION
    else:
        items = kb_items(repo_root, Path(args.kb_root).resolve(), args.kb_version, args.max_chars_per_item)
        index_version = args.index_version or KB_VECTOR_INDEX_VERSION

    if not items:
        raise ValueError(f"no items found for kind={args.kind}")

    embeddings = encode_texts(args.embedding_model_path, [str(item["text"]) for item in items], args.batch_size, args.device)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
        raise ValueError("embedding output shape does not match indexed items")

    npz_path = output / "index.npz"
    np.savez_compressed(npz_path, embeddings=embeddings)

    manifest = {
        "schema_version": "0.1",
        "kind": args.kind,
        "index_version": index_version,
        "index_hash": sha256_file(npz_path),
        "built_at": utc_now(),
        "embedding_model_id": args.embedding_model_id,
        "embedding_model_path": str(Path(args.embedding_model_path).resolve()),
        "loader": "sentence-transformers",
        "embedding_dim": int(embeddings.shape[1]),
        "normalize_embeddings": True,
        "item_count": len(items),
        "source_hash": stable_hash([{k: item.get(k) for k in ("id", "source_path", "content_hash")} for item in items]),
        "chunking": {"max_chars_per_item": args.max_chars_per_item},
        "kb_version": args.kb_version if args.kind == "kb" else "",
        "items": items,
    }
    write_json(output / "manifest.json", manifest)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["raw_corpus", "kb"], required=True)
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--embedding-model-id", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-root", default="sources")
    parser.add_argument("--kb-root", default="kb")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--index-version", default=None)
    parser.add_argument("--max-chars-per-item", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        output = build(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote vector index: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
