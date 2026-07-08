#!/usr/bin/env python3
"""Codex Hook metrics collector for H20 formal experiments.

This script is intentionally dependency-free. Codex invokes it as a command hook
and passes the hook event payload on stdin as JSON. The script records only
side-channel observations; it must not change candidate.py, harness files, or
agent-visible context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-Linux fallback
    fcntl = None

DEFAULT_SNAPSHOT_SECONDS = (120, 240, 480)
DEFAULT_COMPILE_RE = r"(compile|build|triton|jit|nvcc|setup\.py|--stage[=\s]+compile)"
DEFAULT_CORRECTNESS_RE = r"(correctness|pytest|smoke|quick|hidden|validate|--stage[=\s]+(smoke|quick|hidden|correctness))"
DEFAULT_BENCHMARK_RE = r"(benchmark|bench|latency|speedup|profile|nsys|ncu|--stage[=\s]+(benchmark|bench|perf))"
DEFAULT_HARNESS_RE = r"(^|\s)(\./)?run\.sh(\s|$)|harness"


def utc_ms() -> int:
    return int(time.time() * 1000)


def safe_name(value: Any, default: str = "none") -> str:
    raw = str(value if value not in (None, "") else default)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:120]


def read_stdin_event() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def workspace_dir(event: dict[str, Any]) -> Path:
    return Path(os.environ.get("H20_WORKSPACE_DIR") or event.get("cwd") or os.getcwd()).resolve()


def metrics_dir(event: dict[str, Any]) -> Path:
    configured = os.environ.get("H20_METRICS_DIR")
    if configured:
        return Path(configured).resolve()
    return workspace_dir(event) / ".codex_h20_metrics"


def candidate_path(event: dict[str, Any]) -> Path:
    configured = os.environ.get("H20_CANDIDATE_PATH")
    if configured:
        return Path(configured).resolve()
    return workspace_dir(event) / "candidate.py"


def snapshot_seconds() -> list[int]:
    raw = os.environ.get("H20_HOOK_SNAPSHOT_SECONDS")
    if not raw:
        return list(DEFAULT_SNAPSHOT_SECONDS)
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            sec = int(part)
        except ValueError:
            continue
        if sec > 0:
            out.append(sec)
    return sorted(set(out)) or list(DEFAULT_SNAPSHOT_SECONDS)


def agent_key(event: dict[str, Any]) -> str:
    hook_event = event.get("hook_event_name")
    if hook_event in {"SessionStart", "Stop"}:
        return "root"
    return safe_name(event.get("agent_id"), "root")


def read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("ts_ms", utc_ms())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_trace(mdir: Path, record: dict[str, Any]) -> None:
    append_jsonl(mdir / "trace.jsonl", record)


def append_error(event: dict[str, Any], exc: BaseException) -> None:
    try:
        append_jsonl(
            metrics_dir(event) / "hook_errors.jsonl",
            {
                "event": "hook_error",
                "hook_event_name": event.get("hook_event_name"),
                "error": repr(exc),
                "argv": sys.argv,
            },
        )
    except Exception:
        pass


def default_state() -> dict[str, Any]:
    return {
        "created_ms": utc_ms(),
        "session_start_ms": None,
        "session_stop_ms": None,
        "meta": {},
        "agents": {},
        "candidate_seq": 0,
        "candidate_hashes": {},
        "current_candidate_sha256": None,
        "observations": [],
        "timers_started": {},
        "totals": {
            "candidate_edit_attempts": 0,
            "compile_attempts": 0,
            "correctness_runs": 0,
            "benchmark_runs": 0,
            "harness_runs": 0,
        },
    }


def load_state(mdir: Path) -> dict[str, Any]:
    path = mdir / "state.json"
    if not path.exists():
        return default_state()
    try:
        state = json.loads(path.read_text())
    except Exception:
        state = default_state()
        state["state_load_error"] = True
    state.setdefault("meta", {})
    state.setdefault("agents", {})
    state.setdefault("candidate_hashes", {})
    state.setdefault("observations", [])
    state.setdefault("timers_started", {})
    state.setdefault("totals", {})
    for key in ["candidate_edit_attempts", "compile_attempts", "correctness_runs", "benchmark_runs", "harness_runs"]:
        state["totals"].setdefault(key, 0)
    return state


def save_state(mdir: Path, state: dict[str, Any]) -> None:
    mdir.mkdir(parents=True, exist_ok=True)
    state["updated_ms"] = utc_ms()
    tmp = mdir / "state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(mdir / "state.json")


def with_state(mdir: Path, fn: Callable[[dict[str, Any]], Any]) -> Any:
    mdir.mkdir(parents=True, exist_ok=True)
    lock_path = mdir / ".lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        state = load_state(mdir)
        result = fn(state)
        save_state(mdir, state)
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
        return result


def run_meta(event: dict[str, Any]) -> dict[str, Any]:
    ws = workspace_dir(event)
    meta: dict[str, Any] = {
        "workspace": str(ws),
        "run_id": os.environ.get("H20_RUN_ID"),
        "group": os.environ.get("H20_GROUP"),
        "task": os.environ.get("H20_TASK"),
        "candidate_path": str(candidate_path(event)),
    }
    task_json = ws / "task.json"
    task = read_json_if_exists(task_json)
    if isinstance(task, dict):
        meta["task_json"] = {
            key: task.get(key)
            for key in ("task_id", "operator", "category", "dtype", "shape", "group", "seed", "run_id")
            if key in task
        }
    return meta


def ensure_agent(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    key = agent_key(event)
    ag = state["agents"].setdefault(
        key,
        {
            "agent_id": event.get("agent_id"),
            "agent_type": event.get("agent_type"),
            "start_ms": None,
            "stop_ms": None,
            "candidate_edit_attempts": 0,
            "unique_candidate_count": 0,
            "compile_attempts": 0,
            "correctness_runs": 0,
            "benchmark_runs": 0,
            "harness_runs": 0,
            "tool_events": [],
            "candidate_snapshots": [],
            "observations": [],
        },
    )
    ag["agent_id"] = ag.get("agent_id") or event.get("agent_id")
    ag["agent_type"] = ag.get("agent_type") or event.get("agent_type")
    return ag


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def elapsed_fields(state: dict[str, Any], event: dict[str, Any], ts_ms: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    session_start = state.get("session_start_ms") or state.get("created_ms")
    if session_start:
        out["elapsed_from_session_s"] = round((ts_ms - int(session_start)) / 1000.0, 3)
    ag = state.get("agents", {}).get(agent_key(event), {})
    agent_start = ag.get("start_ms")
    if agent_start:
        out["elapsed_from_agent_s"] = round((ts_ms - int(agent_start)) / 1000.0, 3)
    return out


def snapshot_candidate(event: dict[str, Any], source: str, label: str) -> dict[str, Any]:
    mdir = metrics_dir(event)
    cpath = candidate_path(event)
    ts = utc_ms()

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state["meta"].update(run_meta(event))
        ag = ensure_agent(state, event)
        rec: dict[str, Any] = {
            "ts_ms": ts,
            "hook_event_name": event.get("hook_event_name"),
            "agent_key": agent_key(event),
            "agent_id": event.get("agent_id"),
            "agent_type": event.get("agent_type"),
            "source": source,
            "label": label,
            "candidate_path": str(cpath),
            "exists": cpath.exists(),
            **elapsed_fields(state, event, ts),
        }
        if not cpath.exists():
            ag["candidate_snapshots"].append(rec)
            return rec

        digest = sha256_file(cpath)
        rec["sha256"] = digest
        state["current_candidate_sha256"] = digest
        ag["current_candidate_sha256"] = digest

        if digest not in state["candidate_hashes"]:
            state["candidate_seq"] = int(state.get("candidate_seq", 0)) + 1
            seq = int(state["candidate_seq"])
            rel = Path("candidates") / f"{seq:04d}_{safe_name(label)}_{digest[:12]}.py"
            dst = mdir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cpath, dst)
            state["candidate_hashes"][digest] = {
                "seq": seq,
                "path": str(rel),
                "first_seen_ms": ts,
                "first_seen_source": source,
                "first_seen_agent_key": agent_key(event),
            }
            ag["unique_candidate_count"] = int(ag.get("unique_candidate_count", 0)) + 1
            rec["unique"] = True
            rec["candidate_seq"] = seq
            rec["snapshot_file"] = str(rel)
        else:
            known = state["candidate_hashes"][digest]
            rec["unique"] = False
            rec["candidate_seq"] = known.get("seq")
            rec["snapshot_file"] = known.get("path")

        ag["candidate_snapshots"].append(rec)
        return rec

    rec = with_state(mdir, update)
    append_trace(mdir, {"event": "candidate_snapshot", **rec})
    return rec


def record_start(event: dict[str, Any]) -> None:
    mdir = metrics_dir(event)
    ts = utc_ms()
    hook_event = event.get("hook_event_name")

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state["meta"].update(run_meta(event))
        ag = ensure_agent(state, event)
        if hook_event == "SessionStart" and state.get("session_start_ms") is None:
            state["session_start_ms"] = ts
        if hook_event == "SubagentStart" and ag.get("start_ms") is None:
            ag["start_ms"] = ts
        return {
            "hook_event_name": hook_event,
            "agent_key": agent_key(event),
            "agent_id": event.get("agent_id"),
            "agent_type": event.get("agent_type"),
            "ts_ms": ts,
        }

    rec = with_state(mdir, update)
    append_trace(mdir, {"event": "lifecycle_start", **rec})


def record_stop(event: dict[str, Any]) -> None:
    mdir = metrics_dir(event)
    ts = utc_ms()
    hook_event = event.get("hook_event_name")

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state["meta"].update(run_meta(event))
        ag = ensure_agent(state, event)
        if hook_event == "Stop":
            state["session_stop_ms"] = ts
        if hook_event == "SubagentStop":
            ag["stop_ms"] = ts
        return {
            "hook_event_name": hook_event,
            "agent_key": agent_key(event),
            "agent_id": event.get("agent_id"),
            "agent_type": event.get("agent_type"),
            "ts_ms": ts,
        }

    rec = with_state(mdir, update)
    append_trace(mdir, {"event": "lifecycle_stop", **rec})


def event_text(event: dict[str, Any]) -> str:
    tool_input = event.get("tool_input")
    if isinstance(tool_input, dict):
        selected = []
        for key in ("command", "cmd", "patch", "path", "file_path", "filename", "content"):
            value = tool_input.get(key)
            if isinstance(value, str):
                selected.append(value)
        try:
            selected.append(json.dumps(tool_input, ensure_ascii=False))
        except Exception:
            pass
        return "\n".join(selected)
    return str(tool_input or "")


def env_regex(name: str, default: str) -> re.Pattern[str]:
    return re.compile(os.environ.get(name, default), re.IGNORECASE | re.MULTILINE)


def classify_tool(event: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(event.get("tool_name") or "")
    text = event_text(event)
    low = text.lower()
    cpath = candidate_path(event)
    candidate_names = {cpath.name.lower(), "candidate.py"}
    mentions_candidate = any(name in low for name in candidate_names)

    mutating_tool = tool_name in {"apply_patch", "Edit", "Write", "MultiEdit"}
    mutating_bash = bool(
        re.search(
            r"(>|>>|tee\s+|sed\s+-i|perl\s+-pi|mv\s+|cp\s+|write_text|open\s*\(|truncate\s*\(|cat\s+<<|python\s+-\s*<<|apply_patch)",
            low,
            re.IGNORECASE,
        )
    )
    mutates_candidate = mentions_candidate and (mutating_tool or mutating_bash)

    compile_attempt = bool(env_regex("H20_COMPILE_RE", DEFAULT_COMPILE_RE).search(text))
    correctness_run = bool(env_regex("H20_CORRECTNESS_RE", DEFAULT_CORRECTNESS_RE).search(text))
    benchmark_run = bool(env_regex("H20_BENCHMARK_RE", DEFAULT_BENCHMARK_RE).search(text))
    harness_run = bool(env_regex("H20_HARNESS_RE", DEFAULT_HARNESS_RE).search(text))

    return {
        "tool_name": tool_name,
        "tool_use_id": event.get("tool_use_id"),
        "mutates_candidate": mutates_candidate,
        "mentions_candidate": mentions_candidate,
        "compile_attempt": compile_attempt,
        "correctness_run": correctness_run,
        "benchmark_run": benchmark_run,
        "harness_run": harness_run,
        "command_prefix": text[:500],
    }


def record_pre_tool_use(event: dict[str, Any]) -> dict[str, Any]:
    mdir = metrics_dir(event)
    cls = classify_tool(event)

    def inc(state: dict[str, Any], key: str) -> None:
        state["totals"][key] = int(state["totals"].get(key, 0)) + 1
        ag = ensure_agent(state, event)
        ag[key] = int(ag.get(key, 0)) + 1

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state["meta"].update(run_meta(event))
        ag = ensure_agent(state, event)
        if cls["mutates_candidate"]:
            inc(state, "candidate_edit_attempts")
        if cls["compile_attempt"]:
            inc(state, "compile_attempts")
        if cls["correctness_run"]:
            inc(state, "correctness_runs")
        if cls["benchmark_run"]:
            inc(state, "benchmark_runs")
        if cls["harness_run"]:
            inc(state, "harness_runs")
        rec = {"ts_ms": utc_ms(), "agent_key": agent_key(event), **cls}
        ag["tool_events"].append(rec)
        return rec

    rec = with_state(mdir, update)
    append_trace(mdir, {"event": "pre_tool_use", **rec})
    if cls["mutates_candidate"]:
        snapshot_candidate(event, source="PreToolUse", label="before_candidate_edit")
    return rec


def walk_json(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_json(value, child)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from walk_json(value, f"{prefix}[{idx}]")
    else:
        yield prefix, obj


def first_number(pairs: list[tuple[str, Any]], pred: Callable[[str], bool]) -> float | None:
    for key, value in pairs:
        if pred(key.lower()) and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def first_bool(pairs: list[tuple[str, Any]], pred: Callable[[str], bool]) -> bool | None:
    for key, value in pairs:
        if pred(key.lower()) and isinstance(value, bool):
            return bool(value)
    return None


def result_file_payloads(ws: Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for name in ("results.json", "benchmark.json", "profile_summary.json"):
        value = read_json_if_exists(ws / name)
        if value is not None:
            payloads[name] = value
    return payloads


def result_files_are_stale(ws: Path, cpath: Path) -> bool:
    if not cpath.exists():
        return False
    result_paths = [ws / "results.json", ws / "benchmark.json"]
    existing = [p for p in result_paths if p.exists()]
    if not existing:
        return False
    newest_result_mtime = max(p.stat().st_mtime for p in existing)
    return newest_result_mtime + 0.5 < cpath.stat().st_mtime


def parse_result_files(event: dict[str, Any], source: str) -> dict[str, Any] | None:
    ws = workspace_dir(event)
    cpath = candidate_path(event)
    payloads = result_file_payloads(ws)
    if not payloads:
        return None

    pairs: list[tuple[str, Any]] = []
    for payload in payloads.values():
        pairs.extend(list(walk_json(payload)))

    p50_ms = first_number(
        pairs,
        lambda k: k.endswith("latency_p50_ms")
        or k.endswith("p50_ms")
        or k.endswith(".p50")
        or ("latency" in k and "p50" in k),
    )
    speedup_p50 = first_number(pairs, lambda k: "speedup" in k and ("p50" in k or k.endswith("speedup")))
    hidden_pass = first_bool(pairs, lambda k: "hidden" in k and ("pass" in k or "success" in k or "ok" in k))
    correctness_pass = first_bool(pairs, lambda k: "correct" in k and ("pass" in k or "success" in k or "ok" in k))
    legal = first_bool(pairs, lambda k: (k.endswith("legal") or "legality_pass" in k or "is_legal" in k) and "illegal" not in k)
    illegal = first_bool(pairs, lambda k: "illegal" in k and ("candidate" in k or "final" in k or "api" in k or k.endswith("illegal")))
    if legal is None and illegal is not None:
        legal = not illegal

    digest = sha256_file(cpath) if cpath.exists() else None
    stale = result_files_are_stale(ws, cpath)
    legal_for_perf = bool(legal is True and (hidden_pass is True or correctness_pass is True) and p50_ms is not None and not stale)

    obs: dict[str, Any] = {
        "ts_ms": utc_ms(),
        "source": source,
        "hook_event_name": event.get("hook_event_name"),
        "agent_key": agent_key(event),
        "tool_use_id": event.get("tool_use_id"),
        "candidate_sha256": digest,
        "latency_p50_ms": p50_ms,
        "speedup_p50": speedup_p50,
        "hidden_correctness_pass": hidden_pass,
        "correctness_pass": correctness_pass,
        "legal": legal,
        "illegal": illegal,
        "stale_result_files": stale,
        "legal_for_perf": legal_for_perf,
        "result_files": sorted(payloads.keys()),
    }

    mdir = metrics_dir(event)

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state["meta"].update(run_meta(event))
        ag = ensure_agent(state, event)
        state["observations"].append(obs)
        ag["observations"].append(obs)
        return obs

    rec = with_state(mdir, update)
    append_trace(mdir, {"event": "result_observation", **rec})
    return rec


def record_post_tool_use(event: dict[str, Any]) -> None:
    cls = classify_tool(event)
    if cls["mutates_candidate"] or cls["tool_name"] in {"apply_patch", "Edit", "Write", "MultiEdit"}:
        snapshot_candidate(event, source="PostToolUse", label="after_candidate_edit")
    if cls["compile_attempt"] or cls["correctness_run"] or cls["benchmark_run"] or cls["harness_run"]:
        parse_result_files(event, source="PostToolUse")


def timer_started(event: dict[str, Any], scope: str, owner_key: str) -> bool:
    mdir = metrics_dir(event)
    timer_key = f"{scope}:{owner_key}"

    def update(state: dict[str, Any]) -> bool:
        timers = state.setdefault("timers_started", {})
        if timers.get(timer_key):
            return False
        timers[timer_key] = {"started_ms": utc_ms(), "scope": scope, "owner_key": owner_key}
        return True

    return bool(with_state(mdir, update))


def start_timer_process(event: dict[str, Any], scope: str, owner_key: str) -> None:
    if os.environ.get("H20_HOOK_ENABLE_TIMER", "1") == "0":
        return
    if not timer_started(event, scope, owner_key):
        return
    script = Path(__file__).resolve()
    mdir = metrics_dir(event)
    ws = workspace_dir(event)
    env = os.environ.copy()
    env["H20_METRICS_DIR"] = str(mdir)
    env["H20_WORKSPACE_DIR"] = str(ws)
    subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--timer",
            "--scope",
            scope,
            "--owner-key",
            owner_key,
            "--metrics-dir",
            str(mdir),
            "--workspace-dir",
            str(ws),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )


def timer_base_and_stop(mdir: Path, scope: str, owner_key: str) -> tuple[int | None, bool]:
    def read(state: dict[str, Any]) -> tuple[int | None, bool]:
        if scope == "session":
            return state.get("session_start_ms"), bool(state.get("session_stop_ms"))
        ag = state.get("agents", {}).get(owner_key, {})
        return ag.get("start_ms"), bool(ag.get("stop_ms"))

    return with_state(mdir, read)


def timer_main(args: argparse.Namespace) -> int:
    mdir = Path(args.metrics_dir).resolve()
    ws = Path(args.workspace_dir).resolve()
    base_ms, stopped = timer_base_and_stop(mdir, args.scope, args.owner_key)
    if not base_ms or stopped:
        return 0
    base_s = int(base_ms) / 1000.0
    event = {
        "cwd": str(ws),
        "hook_event_name": "Timer",
        "agent_id": args.owner_key if args.scope == "subagent" else None,
        "agent_type": "timer",
    }
    for sec in snapshot_seconds():
        delay = max(0.0, base_s + sec - time.time())
        time.sleep(delay)
        _, stopped = timer_base_and_stop(mdir, args.scope, args.owner_key)
        if stopped:
            append_trace(
                mdir,
                {
                    "event": "timer_snapshot_skipped",
                    "scope": args.scope,
                    "owner_key": args.owner_key,
                    "threshold_s": sec,
                    "reason": "scope_stopped",
                },
            )
            break
        snapshot_candidate(event, source="timer", label=f"{args.scope}_t{sec}s")
    return 0


def summarize_state(event: dict[str, Any]) -> dict[str, Any]:
    mdir = metrics_dir(event)

    def build(state: dict[str, Any]) -> dict[str, Any]:
        observations = state.get("observations", [])
        legal_obs = [
            obs
            for obs in observations
            if obs.get("legal_for_perf") and isinstance(obs.get("latency_p50_ms"), (int, float))
        ]
        any_obs = [obs for obs in observations if isinstance(obs.get("latency_p50_ms"), (int, float)) and not obs.get("stale_result_files")]
        oracle = min(legal_obs, key=lambda obs: float(obs["latency_p50_ms"])) if legal_obs else None
        observed_best = min(any_obs, key=lambda obs: float(obs["latency_p50_ms"])) if any_obs else None
        final_sha = state.get("current_candidate_sha256")
        final_legal = [obs for obs in legal_obs if obs.get("candidate_sha256") == final_sha]
        final_obs = final_legal[-1] if final_legal else None

        session_wall = None
        if state.get("session_start_ms") and state.get("session_stop_ms"):
            session_wall = round((int(state["session_stop_ms"]) - int(state["session_start_ms"])) / 1000.0, 3)

        agents: dict[str, Any] = {}
        for key, ag in state.get("agents", {}).items():
            wall = None
            if ag.get("start_ms") and ag.get("stop_ms"):
                wall = round((int(ag["stop_ms"]) - int(ag["start_ms"])) / 1000.0, 3)
            agents[key] = {
                "agent_id": ag.get("agent_id"),
                "agent_type": ag.get("agent_type"),
                "wall_time_s": wall,
                "candidate_edit_attempts": ag.get("candidate_edit_attempts", 0),
                "unique_candidate_count": ag.get("unique_candidate_count", 0),
                "compile_attempts": ag.get("compile_attempts", 0),
                "correctness_runs": ag.get("correctness_runs", 0),
                "benchmark_runs": ag.get("benchmark_runs", 0),
                "harness_runs": ag.get("harness_runs", 0),
                "snapshot_count": len(ag.get("candidate_snapshots", [])),
            }

        return {
            "generated_ms": utc_ms(),
            "meta": state.get("meta", {}),
            "session_wall_time_s": session_wall,
            "totals": state.get("totals", {}),
            "candidate_count": len(state.get("candidate_hashes", {})),
            "final_candidate_sha256": final_sha,
            "agent_final_legal_p50_ms": final_obs.get("latency_p50_ms") if final_obs else None,
            "agent_final_legal_speedup_p50": final_obs.get("speedup_p50") if final_obs else None,
            "oracle_best_legal_p50_ms": oracle.get("latency_p50_ms") if oracle else None,
            "oracle_best_legal_speedup_p50": oracle.get("speedup_p50") if oracle else None,
            "oracle_best_legal_candidate_sha256": oracle.get("candidate_sha256") if oracle else None,
            "observed_best_unfiltered_p50_ms": observed_best.get("latency_p50_ms") if observed_best else None,
            "observed_best_unfiltered_candidate_sha256": observed_best.get("candidate_sha256") if observed_best else None,
            "agents": agents,
            "notes": [
                "Hook summaries are observational. Official final/oracle p50 should be recomputed by the frozen evaluator from saved candidate snapshots.",
                "legal_for_perf requires legal=true, correctness pass, p50 present, and non-stale result files.",
            ],
        }

    summary = with_state(mdir, build)
    (mdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_trace(mdir, {"event": "summary_written", "summary_path": str(mdir / "summary.json")})
    return summary


def handle_hook_event(event: dict[str, Any]) -> None:
    hook_event = event.get("hook_event_name")
    if hook_event == "SessionStart":
        record_start(event)
        snapshot_candidate(event, source="SessionStart", label="session_start")
        start_timer_process(event, scope="session", owner_key="root")
    elif hook_event == "SubagentStart":
        record_start(event)
        snapshot_candidate(event, source="SubagentStart", label="subagent_start")
        start_timer_process(event, scope="subagent", owner_key=agent_key(event))
    elif hook_event == "PreToolUse":
        record_pre_tool_use(event)
    elif hook_event == "PostToolUse":
        record_post_tool_use(event)
    elif hook_event == "SubagentStop":
        snapshot_candidate(event, source="SubagentStop", label="subagent_stop")
        parse_result_files(event, source="SubagentStop")
        record_stop(event)
        summarize_state(event)
    elif hook_event == "Stop":
        snapshot_candidate(event, source="Stop", label="session_stop")
        parse_result_files(event, source="Stop")
        record_stop(event)
        summarize_state(event)
    else:
        append_trace(metrics_dir(event), {"event": "ignored_hook_event", "hook_event_name": hook_event})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex H20 formal-experiment hook collector")
    parser.add_argument("--timer", action="store_true", help="run as background snapshot timer")
    parser.add_argument("--scope", choices=["session", "subagent"], default="session")
    parser.add_argument("--owner-key", default="root")
    parser.add_argument("--metrics-dir", default="")
    parser.add_argument("--workspace-dir", default="")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.timer:
        if not args.metrics_dir or not args.workspace_dir:
            return 0
        return timer_main(args)

    event: dict[str, Any] = {}
    try:
        event = read_stdin_event()
        handle_hook_event(event)
    except Exception as exc:  # Hooks must not fail the experiment run.
        append_error(event, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
