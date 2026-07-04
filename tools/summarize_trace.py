#!/usr/bin/env python3
"""Summarize an agent/harness trace into state-action-result-diagnosis records."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ecc_utils import read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-events", type=int, default=50)
    args = parser.parse_args()

    rows = read_jsonl(args.trace)
    phase_counts = Counter(str(row.get("phase", "unknown")) for row in rows)
    result_counts = Counter(str(row.get("result") or row.get("status") or "unknown") for row in rows)

    compact: list[dict[str, Any]] = []
    for row in rows[-args.max_events :]:
        compact.append(
            {
                "iteration": row.get("iteration"),
                "phase": row.get("phase"),
                "state_signature": row.get("state_signature") or row.get("candidate_hash") or "",
                "attempted_action": row.get("action_summary") or row.get("action") or "",
                "result": row.get("result") or row.get("status") or "",
                "diagnosis": row.get("diagnosis") or "",
            }
        )

    summary = {
        "trace": str(Path(args.trace)),
        "event_count": len(rows),
        "phase_counts": dict(phase_counts),
        "result_counts": dict(result_counts),
        "first_event": rows[0] if rows else None,
        "last_event": rows[-1] if rows else None,
        "compressed_trajectory": compact,
    }

    write_json(args.out, summary)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
