#!/usr/bin/env python3
"""Shared helpers for the ECC-KB H20 MVP scripts.

The helpers intentionally avoid project-specific services. They operate on files so
that Codex, Claude Code, shell scripts, or a custom controller can use the same
experiment protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root_from(path: str | Path) -> Path:
    path = Path(path).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return Path.cwd().resolve()


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
    return rows


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(data: Any) -> str:
    return sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))


def short_hash(data: Any, length: int = 12) -> str:
    return stable_hash(data)[:length]


def run_cmd(cmd: list[str], cwd: str | Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:  # pragma: no cover - defensive for heterogeneous nodes
        return 1, "", str(exc)


def git_commit(repo_root: str | Path | None = None) -> str:
    code, out, _ = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    return out if code == 0 else "unknown"


def normalize_shape(shape: Any) -> str:
    if isinstance(shape, str):
        return shape
    if isinstance(shape, dict):
        values = [shape.get(k) for k in ("M", "N", "K") if shape.get(k) is not None]
        if values:
            return "x".join(str(v) for v in values)
        return "x".join(f"{k}{v}" for k, v in sorted(shape.items()))
    if isinstance(shape, (list, tuple)):
        return "x".join(str(x) for x in shape)
    return str(shape)


def shape_bucket(shape: Any) -> str:
    values: list[int] = []
    if isinstance(shape, dict):
        values = [int(v) for v in shape.values() if isinstance(v, int)]
    elif isinstance(shape, (list, tuple)):
        values = [int(v) for v in shape if isinstance(v, int)]
    if not values:
        return "unknown"
    max_dim = max(values)
    if max_dim <= 256:
        scale = "small"
    elif max_dim <= 1024:
        scale = "medium"
    elif max_dim <= 4096:
        scale = "large"
    else:
        scale = "xlarge"
    non_power_two = any(v > 0 and (v & (v - 1)) != 0 for v in values)
    return f"{scale}{'_nonpow2' if non_power_two else '_pow2'}"


def safe_id(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def iter_json_files(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".json":
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("*.json")))
    return sorted(set(found))


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def toolchain_from_backend_fingerprint(backend: dict[str, Any]) -> dict[str, str]:
    toolchain = backend.get("toolchain") or backend.get("software", {}).get("toolchain") or {}
    return {str(k): str(v) for k, v in toolchain.items()}


def backend_required_fields(backend: dict[str, Any]) -> dict[str, Any]:
    return {
        "vendor": str(backend.get("vendor", "unknown")),
        "backend": str(backend.get("backend", "unknown")),
        "device_name": str(backend.get("device_name") or backend.get("hardware", {}).get("device_name", "unknown")),
        "device_class": str(backend.get("device_class") or backend.get("hardware", {}).get("device_class", "runtime_detected")),
        "toolchain": toolchain_from_backend_fingerprint(backend),
    }


def print_error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
