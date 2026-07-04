#!/usr/bin/env python3
"""Prepare or evaluate one agent-neutral H20 experiment session.

This script does not invoke a specific Coding Agent. It creates a workspace with
`task.json`, `context_packet.json`, `candidate.py`, `notes.md`, and `run.sh`.
After a human or external agent edits `candidate.py`, run the script again with
`--run-harness` or execute `run.sh` inside the workspace.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import append_jsonl, read_json, sha256_file, utc_now, write_json  # noqa: E402

SKELETONS = {
    "softmax": """\"\"\"Candidate kernel for row softmax. Interface: candidate(x) -> y. Do not call torch.softmax.\"\"\"\n\ndef candidate(x):\n    raise NotImplementedError(\"agent must implement row softmax\")\n""",
    "reduction": """\"\"\"Candidate kernel for row reduction. Interface: candidate(x, reduce_op) -> y. Do not call torch.sum or torch.max.\"\"\"\n\ndef candidate(x, reduce_op):\n    raise NotImplementedError(\"agent must implement row reduction\")\n""",
    "layernorm": """\"\"\"Candidate kernel for LayerNorm forward. Interface: candidate(x, gamma, beta, eps) -> y.\"\"\"\n\ndef candidate(x, gamma, beta, eps):\n    raise NotImplementedError(\"agent must implement layernorm forward\")\n""",
    "matmul": """\"\"\"Candidate kernel for matmul. Interface: candidate(a, b) -> c. Do not call torch.matmul.\"\"\"\n\ndef candidate(a, b):\n    raise NotImplementedError(\"agent must implement matmul\")\n""",
}


def infer_hidden_path(task_path: Path) -> Path | None:
    hidden = task_path.parent / "_hidden" / f"{task_path.stem}.hidden.json"
    return hidden if hidden.exists() else None


def write_prompt(workspace: Path, task: dict[str, Any], group: str, phase: str) -> Path:
    prompt = workspace / "agent_prompt.md"
    prompt.write_text(
        f"""# H20 Kernel Generation Task

Edit `candidate.py` for this ECC-KB experiment.

Group: `{group}`
Phase: `{phase}`
Task: `{task.get('task_id')}`
Operator: `{task.get('op_family')}` / `{task.get('op_name')}`
Shape: `{task.get('shape')}`
DType: `{task.get('dtype')}`
Interface: `{task.get('op_spec', {}).get('candidate_interface')}`

Rules:
- Modify only `candidate.py` and optionally `notes.md`.
- Read `context_packet.json` before editing.
- Preserve the declared interface.
- Pass hidden correctness before performance tuning.
- Do not hardcode public shapes.
""",
        encoding="utf-8",
    )
    return prompt


def write_run_sh(workspace: Path, harness: Path, hidden_path: Path | None, warmup: int | None, repeats: int | None) -> None:
    cmd = ["python", str(harness), "--task", "task.json", "--candidate", "candidate.py", "--out-dir", "."]
    if hidden_path is not None:
        cmd.extend(["--hidden-tests", str(hidden_path.resolve())])
    if warmup is not None:
        cmd.extend(["--warmup", str(warmup)])
    if repeats is not None:
        cmd.extend(["--repeats", str(repeats)])
    path = workspace / "run.sh"
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(cmd) + "\n", encoding="utf-8")
    path.chmod(0o755)


def prepare_workspace(args: argparse.Namespace, phase: str) -> Path:
    repo_root = Path.cwd()
    task_path = Path(args.task).resolve()
    backend_path = Path(args.backend).resolve()
    hidden_path = Path(args.hidden_tests).resolve() if args.hidden_tests else infer_hidden_path(task_path)
    workspace = Path(args.out).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    task = read_json(task_path)
    shutil.copy2(task_path, workspace / "task.json")

    candidate = workspace / "candidate.py"
    if not candidate.exists():
        candidate.write_text(SKELETONS.get(task.get("op_family"), "def candidate(*args):\n    raise NotImplementedError()\n"), encoding="utf-8")
    if not (workspace / "notes.md").exists():
        (workspace / "notes.md").write_text("# Agent notes\n", encoding="utf-8")

    context_path = workspace / "context_packet.json"
    retrieve_cmd = [
        sys.executable,
        str((repo_root / args.retrieve_script).resolve()),
        "--task", str(workspace / "task.json"),
        "--backend", str(backend_path),
        "--group", args.group,
        "--phase", phase,
        "--kb-version", args.kb_version,
        "--out", str(context_path),
    ]
    subprocess.run(retrieve_cmd, cwd=str(repo_root), check=True)
    write_prompt(workspace, task, args.group, phase)
    write_run_sh(workspace, (repo_root / args.harness).resolve(), hidden_path, args.warmup, args.repeats)
    return workspace


def run_harness(args: argparse.Namespace, workspace: Path) -> int:
    repo_root = Path.cwd()
    task_path = workspace / "task.json"
    hidden_path = Path(args.hidden_tests).resolve() if args.hidden_tests else infer_hidden_path(Path(args.task).resolve())
    cmd = [sys.executable, str((repo_root / args.harness).resolve()), "--task", str(task_path), "--candidate", str(workspace / "candidate.py"), "--out-dir", str(workspace), "--seed", str(args.seed)]
    if hidden_path is not None:
        cmd.extend(["--hidden-tests", str(hidden_path)])
    if args.warmup is not None:
        cmd.extend(["--warmup", str(args.warmup)])
    if args.repeats is not None:
        cmd.extend(["--repeats", str(args.repeats)])
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    results = read_json(workspace / "results.json", default={}) if (workspace / "results.json").exists() else {}
    event = {
        "timestamp": utc_now(),
        "iteration": args.iteration,
        "phase": args.phase,
        "action_summary": f"harness_rc={proc.returncode}",
        "candidate_hash": sha256_file(workspace / "candidate.py"),
        "compile": results.get("compile", {}),
        "correctness": results.get("correctness", {}),
        "benchmark": results.get("benchmark", {}),
        "profile_summary": results.get("profile_summary", {}),
        "result": "keep_candidate" if results.get("final_decision") == "KEEP" else ("discard_candidate" if results.get("final_decision") == "DISCARD" else "correctness_fail"),
        "diagnosis": results.get("diagnosis", ""),
        "harness_stdout_tail": proc.stdout[-2000:],
        "harness_stderr_tail": proc.stderr[-2000:],
        "timing": {"harness_s": time.time() - started},
    }
    append_jsonl(workspace / "trace.jsonl", event)
    if results:
        results.setdefault("cost", {})
        results["cost"]["iterations"] = args.iteration + 1
        write_json(workspace / "results.json", results)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["A0_no_kb", "A1_plain_rag", "A2_rulebook", "A3_ecc_kb"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", default="generate")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--run-harness", action="store_true")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--hidden-tests", default=None)
    parser.add_argument("--retrieve-script", default="tools/retrieve_context.py")
    parser.add_argument("--harness", default="experiments/h20/harness.py")
    args = parser.parse_args()

    workspace = prepare_workspace(args, args.phase)
    if args.run_harness:
        return run_harness(args, workspace)
    print(f"Workspace prepared: {workspace}")
    print("Edit candidate.py, then run:")
    print(f"  cd {workspace} && ./run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
