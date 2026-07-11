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

PROMPT_TEMPLATE = Path(__file__).with_name("agent_prompt_template.md")

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


def template_section(name: str) -> str:
    text = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    start = f"<!-- section:{name} -->"
    end = "<!-- endsection -->"
    if start not in text:
        raise KeyError(f"missing prompt template section: {name}")
    section = text.split(start, 1)[1].split(end, 1)[0]
    return section.strip()


def render_section(name: str, values: dict[str, Any] | None = None) -> str:
    section = template_section(name)
    return section.format_map(values or {})


def workflow_protocol() -> str:
    return render_section("workflow")


def candidate_skeleton(task: dict[str, Any]) -> str:
    kbx = task.get("kernelbenchx")
    if isinstance(kbx, dict) and kbx.get("entrypoint") and kbx.get("signature"):
        entrypoint = str(kbx["entrypoint"])
        signature = str(kbx["signature"])
        return (
            f'"""Candidate kernel for H20 task {task.get("task_id")}.\n\n'
            f"Required entrypoint: {signature}\n"
            '"""\n\n'
            f"def {signature}:\n"
            f"    raise NotImplementedError(\"agent must implement {entrypoint}\")\n"
        )
    return SKELETONS.get(task.get("op_family"), "def candidate(*args):\n    raise NotImplementedError()\n")


def prompt_interface(task: dict[str, Any]) -> str:
    kbx = task.get("kernelbenchx")
    if isinstance(kbx, dict) and kbx.get("entrypoint") and kbx.get("signature"):
        return str(kbx["signature"])
    return str(task.get("op_spec", {}).get("candidate_interface"))


def render_prompt_sections(sections: list[str]) -> str:
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def posthoc_portability_protocol(task: dict[str, Any]) -> str:
    op_name = str(task.get("op_name") or task.get("op_family") or "operator")
    impl_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in op_name).strip("_") or "operator"
    kbx = task.get("kernelbenchx")
    if isinstance(kbx, dict) and kbx.get("entrypoint") and kbx.get("signature") and kbx.get("task_file"):
        entrypoint = str(kbx["entrypoint"])
        signature = str(kbx["signature"])
        task_file = str(kbx["task_file"])
        return render_section("kbx_portability", {"entrypoint": entrypoint, "signature": signature, "task_file": task_file})
    return render_section("legacy_portability", {"impl_name": impl_name})


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
        return render_section("a0_retrieval")

    if group == "A1_raw_corpus_vector_rag":
        return render_section(
            "a1_retrieval",
            {"phase_commands": phase_commands(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, kb_version)},
        )

    if group == "A2_kb_vector_rag":
        return render_section(
            "a2_retrieval",
            {"phase_commands": phase_commands(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, kb_version)},
        )

    return render_section(
        "a3_retrieval",
        {"phase_commands": phase_commands(retrieve_script, backend_path, kb_root, source_root, raw_vector_index, kb_vector_index, embedding_model_path, group, kb_version)},
    )


def output_protocol(group: str) -> str:
    if group != "A0_prompt":
        return render_section("retrieval_outputs")

    return render_section("a0_outputs")


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
    workflow = workflow_protocol()
    posthoc_portability = posthoc_portability_protocol(task)
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
        render_prompt_sections(
            [
                render_section(
                    "header",
                    {
                        "group": group,
                        "phase": phase,
                        "task_id": task.get("task_id"),
                        "op_family": task.get("op_family"),
                        "op_name": task.get("op_name"),
                        "shape": task.get("shape"),
                        "dtype": task.get("dtype"),
                        "interface": prompt_interface(task),
                    },
                ),
                render_section("background"),
                render_section("rules"),
                workflow,
                posthoc_portability,
                protocol,
                outputs,
            ]
        ),
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
        candidate.write_text(candidate_skeleton(task), encoding="utf-8")
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
    cmd = [sys.executable, str((repo_root / args.harness).resolve()), "--task", str(task_path), "--candidate", str(workspace / "candidate.py"), "--out-dir", str(workspace), "--run", str(args.run), "--require-cuda"]
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
    parser.add_argument("--run", type=int, default=None, help="Run index; also used for harness input reproducibility.")
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
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
    if args.run is None:
        args.run = 0 if args.seed is None else args.seed

    workspace = prepare_workspace(args, args.phase)
    if args.run_harness:
        return run_harness(args, workspace)
    print(f"Workspace prepared: {workspace}")
    print("Edit candidate.py, then run:")
    print(f"  cd {workspace} && ./run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
