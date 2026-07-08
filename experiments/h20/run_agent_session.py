#!/usr/bin/env python3
"""Prepare or evaluate one agent-neutral H20 experiment session.

This script does not invoke a specific Coding Agent. It creates a workspace with
`task.json`, `agent_prompt.md`, `candidate.py`, `run.sh`, and `context_packets/`.
After a human or external agent edits `candidate.py`, run the script again with
`--run-harness` or execute `run.sh` inside the workspace.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import append_jsonl, read_json, sha256_file, utc_now, write_json  # noqa: E402

GROUP_ALIASES = {
    "A0_no_kb": "A0_prompt",
    "A1_plain_rag": "A1_raw_corpus_vector_rag",
    "A1_raw_corpus_rag": "A1_raw_corpus_vector_rag",
    "A2_rulebook": "A2_kb_vector_rag",
    "A2_kb_plain_rag": "A2_kb_vector_rag",
}

GROUPS = ["A0_prompt", "A1_raw_corpus_vector_rag", "A2_kb_vector_rag", "A3_ecc_kb", *GROUP_ALIASES]

SKELETONS = {
    "pointwise": """\"\"\"Candidate kernel for a fused pointwise task. See task.json for the exact interface.\"\"\"\n\ndef candidate(*args):\n    raise NotImplementedError(\"agent must implement the fused pointwise operator declared in task.json\")\n""",
    "softmax": """\"\"\"Candidate kernel for row softmax. Interface: candidate(x) -> y.\"\"\"\n\ndef candidate(x):\n    raise NotImplementedError(\"agent must implement row softmax\")\n""",
    "reduction": """\"\"\"Candidate kernel for row reduction. Interface: candidate(x, reduce_op) -> y.\"\"\"\n\ndef candidate(x, reduce_op):\n    raise NotImplementedError(\"agent must implement row reduction\")\n""",
    "layernorm": """\"\"\"Candidate kernel for LayerNorm forward. Interface: candidate(x, gamma, beta, eps) -> y.\"\"\"\n\ndef candidate(x, gamma, beta, eps):\n    raise NotImplementedError(\"agent must implement layernorm forward\")\n""",
    "matmul": """\"\"\"Candidate kernel for matmul. Interface: candidate(a, b) -> c.\"\"\"\n\ndef candidate(a, b):\n    raise NotImplementedError(\"agent must implement matmul\")\n""",
    "layout": """\"\"\"Candidate kernel for a layout/irregular-memory task. See task.json for the exact interface.\"\"\"\n\ndef candidate(*args):\n    raise NotImplementedError(\"agent must implement the layout operator declared in task.json\")\n""",
}


FORBIDDEN_TARGET_API_RULE = (
    "Do not bypass the task by calling any library API that directly implements "
    "the target operator or an equivalent fused target operator from candidate.py. "
    "Such calls are invalid for this experiment; implement the operator with "
    "Triton/CUDA-style kernels or lower-level primitive operations instead."
)


def infer_hidden_path(task_path: Path) -> Path | None:
    hidden = task_path.parent / "_hidden" / f"{task_path.stem}.hidden.json"
    return hidden if hidden.exists() else None


def retrieval_command(
    retrieve_script: Path,
    backend_path: Path,
    kb_root: Path,
    source_root: Path,
    raw_vector_index: Path,
    kb_vector_index: Path,
    embedding_model_path: str | None,
    group: str,
    phase: str,
    kb_version: str,
    out_name: str,
) -> str:
    parts = [
        "python",
        str(retrieve_script),
        "--task",
        "task.json",
        "--backend",
        str(backend_path),
        "--group",
        group,
        "--phase",
        phase,
        "--kb-version",
        kb_version,
        "--kb-root",
        str(kb_root),
        "--source-root",
        str(source_root),
        "--raw-vector-index",
        str(raw_vector_index),
        "--kb-vector-index",
        str(kb_vector_index),
    ]
    if embedding_model_path:
        parts.extend(["--embedding-model-path", embedding_model_path])
    parts.extend(["--query", "\"<write a concrete query for the current problem and phase>\""])
    parts.extend(["--out", f"context_packets/{out_name}"])
    return " ".join(parts)


def phase_commands(
    retrieve_script: Path,
    backend_path: Path,
    kb_root: Path,
    source_root: Path,
    raw_vector_index: Path,
    kb_vector_index: Path,
    embedding_model_path: str | None,
    group: str,
    kb_version: str,
) -> str:
    return "\n".join(
        f"- `{phase_name}`: `{retrieval_command(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, phase_name, kb_version, f'iter1_{phase_name}.json')}`"
        for phase_name in ("generate", "correctness_repair", "performance_optimize", "autotune")
    )


def background_protocol() -> str:
    return """Background:
- This is an H20 operator-kernel generation experiment.
- Your job is to implement the public operator specification in `candidate.py`, then use the local harness to make it compile, pass correctness checks, and improve latency.
- The experiment compares how different context sources affect the same agent workflow. Keep the workflow below unchanged; only use the retrieval source allowed by your group.
- The phase names describe your current intent:
  - `generate`: create or substantially rewrite the initial implementation.
  - `correctness_repair`: fix compile, smoke, quick, or hidden correctness failures.
  - `performance_optimize`: improve latency after correctness passes.
  - `autotune`: try small launch/configuration variations after a correct implementation exists.
- `./run.sh` is the measurement interface. Treat its stdout and output files as the evidence for phase changes and final claims.
- `task.json` is public task information. Hidden tests and harness internals are not part of your allowed context."""


def workflow_protocol() -> str:
    return """Workflow:
- All experiment groups use this same phase-control loop. The only intended difference is the retrieval protocol below.
- You choose your own phase: `generate`, `correctness_repair`, `performance_optimize`, or `autotune`.
- Before the first substantial edit, enter `generate` and follow this group's retrieval protocol.
- Use staged harness runs whenever you need measurement evidence from the GPU:
  - `./run.sh --stage compile` for import/build feedback.
  - `./run.sh --stage smoke`, `./run.sh --stage quick`, or `./run.sh --stage hidden` for correctness feedback.
  - `./run.sh --stage benchmark` after hidden correctness passes and you need latency evidence.
  - `./run.sh` is the full local check when you need compile, all correctness suites, and benchmark together.
- After every `./run.sh` run, classify the next phase from the harness output:
  - compile, smoke, quick, or hidden failure -> `correctness_repair`
  - correctness passes but latency or profile summary is weak -> `performance_optimize`
  - correctness passes and only launch/configuration knobs remain -> `autotune`
- Before editing for a selected phase, follow this group's retrieval protocol for that phase.
- When retrieval is allowed, write a concrete query for the current problem before calling retrieval. The query should describe what you need now, such as the failing symptom, operator, dtype/shape, suspected cause, performance bottleneck, or tuning knob.
- If retrieval is allowed for this group, do not treat retrieval as optional background reading; preserve the packet as evidence even when it contains no matches.
- Fix correctness failures before performance tuning. Optimize and autotune only after correctness passes."""


def tuning_protocol() -> str:
    return """Common tuning budget:
- After correctness passes, test small launch/configuration variants when they apply to your implementation.
- For a Triton row-wise kernel with `num_warps`, test `8`, `4`, `2`, and `1` when those values compile for the chosen block size.
- Keep the fastest correct variant by the harness p50 result. If a value cannot compile or is clearly inapplicable, record that through `./run.sh` output or a brief code comment in `candidate.py` only if needed.
- Do not continue random tuning after this fixed budget unless correctness is still failing."""


def group_retrieval_protocol(
    retrieve_script: Path,
    backend_path: Path,
    kb_root: Path,
    source_root: Path,
    raw_vector_index: Path,
    kb_vector_index: Path,
    embedding_model_path: str | None,
    group: str,
    kb_version: str,
) -> str:
    if group == "A0_prompt":
        return """Retrieval protocol:
- This is the no-retrieval baseline.
- Do not call `tools/retrieve_context.py` or read `kb/`, `sources/raw_corpus/`, or `sources/raw_archive/`.
- Use only `task.json`, `candidate.py`, `run.sh`, harness output files, and your own reasoning."""

    if group == "A1_raw_corpus_vector_rag":
        return f"""Retrieval protocol:
- This group may use only raw-corpus vector RAG.
- Retrieval is mandatory at phase entry: call retrieval with `--group A1_raw_corpus_vector_rag` before acting in each selected phase.
- You must replace the placeholder after `--query` with your own specific query for this phase/problem. Do not use a generic phase-only query.
- Do not skip retrieval just because you already have an implementation or tuning idea.
- Do not read `kb/` directly and do not call retrieval with A2 or A3 groups.
- Save retrieved packets under `context_packets/`.
- Phase-specific retrieval commands from this workspace:
{phase_commands(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, kb_version)}"""

    if group == "A2_kb_vector_rag":
        return f"""Retrieval protocol:
- This group may use only vector RAG over flattened KB units.
- Retrieval is mandatory at phase entry: call retrieval with `--group A2_kb_vector_rag` before acting in each selected phase.
- You must replace the placeholder after `--query` with your own specific query for this phase/problem. Do not use a generic phase-only query.
- Do not skip retrieval just because you already have an implementation or tuning idea.
- Do not read `sources/raw_corpus/` or `sources/raw_archive/` directly and do not call retrieval with A1 or A3 groups.
- Treat retrieved KB units as ordinary text snippets, not as structured ECC capsules.
- Save retrieved packets under `context_packets/`.
- Phase-specific retrieval commands from this workspace:
{phase_commands(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, kb_version)}"""

    return f"""Retrieval protocol:
- This group uses ECC-KB structured context packets.
- Retrieval is mandatory at ECC phase entry: call retrieval with `--group A3_ecc_kb` before acting in each selected phase.
- You must replace the placeholder after `--query` with your own specific query for this phase/problem. Do not use a generic phase-only query.
- Do not skip retrieval just because you already have an implementation or tuning idea.
- Use the structured packet fields such as `must_obey`, `recommended_actions`, `anti_actions`, `validation_plan`, `stop_conditions`, and `evidence_refs`.
- Do not call retrieval with A1 or A2 groups, and do not browse raw corpus files directly.
- Save retrieved packets under `context_packets/`.
- Phase-specific commands from this workspace:
{phase_commands(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, kb_version)}"""


def output_protocol(group: str) -> str:
    if group != "A0_prompt":
        return """Outputs:
- Final implementation: `candidate.py`.
- Required retrieved context evidence: `context_packets/*.json` for every phase you enter.
- Measurement outputs from `./run.sh`: `results.json`, `compile.log`, `correctness.log`, `benchmark.json`, `profile_summary.json`."""

    return """Outputs:
- Final implementation: `candidate.py`.
- No retrieval output is expected for this no-retrieval baseline.
- Measurement outputs from `./run.sh`: `results.json`, `compile.log`, `correctness.log`, `benchmark.json`, `profile_summary.json`."""


def write_prompt(
    workspace: Path,
    task: dict[str, Any],
    group: str,
    phase: str,
    backend_path: Path,
    retrieve_script: Path,
    kb_version: str,
    raw_vector_index: Path,
    kb_vector_index: Path,
    embedding_model_path: str | None,
) -> Path:
    prompt = workspace / "agent_prompt.md"
    repo_root = Path.cwd()
    kb_root = (repo_root / "kb").resolve()
    source_root = (repo_root / "sources").resolve()
    background = background_protocol()
    workflow = workflow_protocol()
    tuning = tuning_protocol()
    protocol = group_retrieval_protocol(
        retrieve_script,
        backend_path,
        kb_root,
        source_root,
        raw_vector_index,
        kb_vector_index,
        embedding_model_path,
        group,
        kb_version,
    )
    outputs = output_protocol(group)
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

{background}

Rules:
- Read `task.json` first. It is the complete public task specification.
- Modify only `candidate.py`. Keep retrieved context packets under `context_packets/`.
- Preserve the declared interface.
- Do not hardcode public shapes.
- {FORBIDDEN_TARGET_API_RULE}
- Do not read hidden test files or modify the harness.

{workflow}

{tuning}

{protocol}

{outputs}
""",
        encoding="utf-8",
    )
    return prompt


def write_run_sh(
    workspace: Path,
    harness: Path,
    hidden_path: Path | None,
    warmup: int | None,
    repeats: int | None,
    conda_env: str | None,
) -> None:
    cmd = ["python", str(harness), "--task", "task.json", "--candidate", "candidate.py", "--out-dir", ".", "--require-cuda"]
    if hidden_path is not None:
        cmd.extend(["--hidden-tests", str(hidden_path.resolve())])
    if warmup is not None:
        cmd.extend(["--warmup", str(warmup)])
    if repeats is not None:
        cmd.extend(["--repeats", str(repeats)])
    prelude = "#!/usr/bin/env bash\nset -euo pipefail\n"
    if conda_env:
        prelude += (
            "source /data/miniconda3/etc/profile.d/conda.sh\n"
            f"conda activate {conda_env}\n"
        )
    path = workspace / "run.sh"
    path.write_text(
        prelude
        + " ".join(cmd)
        + ' "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def prepare_workspace(args: argparse.Namespace, phase: str) -> Path:
    repo_root = Path.cwd()
    task_path = Path(args.task).resolve()
    backend_path = Path(args.backend).resolve()
    hidden_path = Path(args.hidden_tests).resolve() if args.hidden_tests else infer_hidden_path(task_path)
    workspace = Path(args.out).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    task = read_json(task_path)
    task["agent_allowed_files"] = ["candidate.py", "context_packets/"]
    task["agent_forbidden_files"] = ["reference.py", "hidden_tests.json", "experiments/h20/harness.py", "kb/", "sources/raw_corpus/", "sources/raw_archive/"]
    group = GROUP_ALIASES.get(args.group, args.group)
    write_json(workspace / "task.json", task)

    candidate = workspace / "candidate.py"
    if not candidate.exists():
        candidate.write_text(SKELETONS.get(task.get("op_family"), "def candidate(*args):\n    raise NotImplementedError()\n"), encoding="utf-8")
    (workspace / "context_packets").mkdir(exist_ok=True)

    retrieve_script = (repo_root / args.retrieve_script).resolve()
    raw_vector_index = (repo_root / args.raw_vector_index).resolve()
    kb_vector_index = (repo_root / args.kb_vector_index).resolve()
    write_prompt(
        workspace,
        task,
        group,
        phase,
        backend_path,
        retrieve_script,
        args.kb_version,
        raw_vector_index,
        kb_vector_index,
        args.embedding_model_path,
    )
    write_run_sh(workspace, (repo_root / args.harness).resolve(), hidden_path, args.warmup, args.repeats, args.conda_env)
    return workspace


def run_harness(args: argparse.Namespace, workspace: Path) -> int:
    repo_root = Path.cwd()
    task_path = workspace / "task.json"
    hidden_path = Path(args.hidden_tests).resolve() if args.hidden_tests else infer_hidden_path(Path(args.task).resolve())
    cmd = [sys.executable, str((repo_root / args.harness).resolve()), "--task", str(task_path), "--candidate", str(workspace / "candidate.py"), "--out-dir", str(workspace), "--seed", str(args.seed), "--require-cuda"]
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
    parser.add_argument("--group", choices=GROUPS, required=True)
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
    parser.add_argument("--raw-vector-index", default="artifacts/indexes/raw_corpus_vector_v0")
    parser.add_argument("--kb-vector-index", default="artifacts/indexes/kb_vector_v0")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--conda-env", default="op_kb_dxd", help="Conda environment activated by generated run.sh. Use an empty value to disable activation.")
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
