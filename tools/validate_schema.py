#!/usr/bin/env python3
"""Validate ECC-KB JSON files against a JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecc_utils import iter_json_files, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, help="Path to JSON Schema.")
    parser.add_argument("--file", action="append", default=[], help="JSON file to validate. Can be repeated.")
    parser.add_argument("--glob", action="append", default=[], help="Glob expression for JSON files.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        import jsonschema  # type: ignore
    except Exception:
        print(
            "error: jsonschema is required. Install with: python -m pip install jsonschema",
            file=sys.stderr,
        )
        return 2

    schema = read_json(Path(args.schema))

    files: list[Path] = [Path(p) for p in args.file]
    for pattern in args.glob:
        files.extend(sorted(Path().glob(pattern)))
    files = iter_json_files(files)

    if not files:
        print("error: no JSON files matched", file=sys.stderr)
        return 2

    validator = jsonschema.Draft202012Validator(schema)
    failures = 0

    for path in files:
        try:
            data = read_json(path)
            errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        except json.JSONDecodeError as exc:
            failures += 1
            print(f"FAIL {path}: invalid JSON: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            failures += 1
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            continue

        if errors:
            failures += 1
            print(f"FAIL {path}", file=sys.stderr)
            for error in errors[:20]:
                loc = "/".join(str(p) for p in error.path) or "<root>"
                print(f"  - {loc}: {error.message}", file=sys.stderr)
            if len(errors) > 20:
                print(f"  ... {len(errors) - 20} more errors", file=sys.stderr)
        elif not args.quiet:
            print(f"PASS {path}")

    if failures:
        print(f"validated={len(files)} failed={failures}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"validated={len(files)} failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
