#!/usr/bin/env python3
"""Validate ECC-KB JSON files against a JSON Schema."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        paths.extend(Path(p) for p in matches if Path(p).is_file())
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, help="JSON schema path.")
    parser.add_argument("--glob", action="append", default=[], help="File glob to validate. Can be repeated.")
    parser.add_argument("paths", nargs="*", help="Explicit JSON files to validate.")
    args = parser.parse_args()

    try:
        import jsonschema  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install jsonschema first: python -m pip install jsonschema") from exc

    schema = load_json(args.schema)
    files = iter_paths(args.glob + args.paths)
    if not files:
        raise SystemExit("No files matched.")

    validator = jsonschema.Draft202012Validator(schema)
    failed = 0
    for path in files:
        try:
            doc = load_json(path)
            errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
            if errors:
                failed += 1
                print(f"FAIL {path}")
                for error in errors[:20]:
                    loc = "/".join(str(p) for p in error.path) or "<root>"
                    print(f"  - {loc}: {error.message}")
            else:
                print(f"PASS {path}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {path}: {exc}")

    if failed:
        print(f"{failed} file(s) failed schema validation.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(files)} file(s).")


if __name__ == "__main__":
    main()
