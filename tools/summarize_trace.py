#!/usr/bin/env python3
"""Compress an Agent trace into state-action-result-diagnosis records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def classify_result(event: dict[str, Any]) -> str:
    text = json.dumps(event, ensure_ascii=False).lower()
    if "compile" in text and ("fail" in text or "error" in text):
        return "compile_fail"
    if "correctness" in text and ("fail" in text or "mismatch" in text):
        return "correctness_fail"
    if "hidden" in text and "pass" in text:
        return "keep_candidate"
    if "timeout" in text:
        return "timeout"
    return str(event.get("result") or event.get("status") or "needs_more_evidence")


def summarize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for i, event in enumerate(events):
        summary.append(
            {
                "iteration": int(event.get("iteration", i)),
                "phase": event.get("phase", "unknown"),
                "action_summary": event.get("action_summary") or event.get("action") or event.get("message") or "",
                "candidate_hash": event.get("candidate_hash"),
                "result": classify_result(event),
                "diagnosis": event.get("diagnosis") or event.get("error") or event.get("summary") or "",
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    events = read_jsonl(Path(args.trace))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summarize(events), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
