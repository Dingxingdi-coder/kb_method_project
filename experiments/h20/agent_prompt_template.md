<!--
This file is the source for generated workspace agent_prompt.md files.
Read sections from top to bottom to review the final prompt order.

Curly-brace placeholders are runtime values rendered by run_agent_session.py.
The script selects exactly one portability section, one retrieval section, and
one output section for each workspace.
-->

<!-- section:header -->
# H20 Kernel Generation Task
<!-- section:background -->
Background:
- You are a specialist in kernel generation.
- Your job is to implement the public operator specification in `candidate.py`, then use the local harness to make it compile, pass correctness checks, and improve latency.
- Keep the workflow below unchanged; only use the retrieval source allowed by your group.
- The phase names describe your current intent:
  - `generate`: create or substantially rewrite the initial implementation.
  - `correctness_repair`: fix compile, smoke, quick, or hidden correctness failures.
  - `performance_optimize`: improve latency after correctness passes.
  - `autotune`: try launch/configuration variations after a correct implementation exists.
- `./run.sh` is the measurement interface. Treat its stdout and output files as the evidence for phase changes and final claims.
- `task.json` is public task information. Hidden tests and harness internals are not part of your allowed context.
<!-- endsection -->

Edit `candidate.py` for this ECC-KB experiment.

Group: `{group}`
Phase: `{phase}`
Task: `{task_id}`
Operator: `{op_family}` / `{op_name}`
Shape: `{shape}`
DType: `{dtype}`
Interface: `{interface}`
<!-- endsection -->

<!-- section:rules -->
Rules:
- Read `task.json` first. It is the complete public task specification.
- Modify only `candidate.py`. Keep retrieved context packets under `context_packets/`.
- Preserve the declared interface.
- Do not hardcode public shapes.
- Do not bypass the task by calling any library API that directly implements the target operator or an equivalent fused target operator from candidate.py. Such calls are invalid for this experiment; implement the operator with Triton/CUDA-style kernels or lower-level primitive operations instead.
- Do not read hidden test files or modify the harness.
<!-- endsection -->

<!-- section:workflow -->
Workflow:
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
- Fix correctness failures before performance tuning. Optimize and autotune only after correctness passes.
<!-- endsection -->

<!-- section:kbx_portability -->
Submission interface:
- `candidate.py` remains the submission file for this H20 workspace.
- This task requires the top-level callable `def {signature}` in `candidate.py`.
- Follow the public callable semantics in `task.json`.
- The function name must remain `{entrypoint}`; do not expose only `kernel_function`, because the H20 harness and exporter validate the explicit task entrypoint.
- Do not add or rely on an extra `candidate(...)` wrapper for this task.
- Keep helper kernels and helper functions at module top level rather than nested inside `{entrypoint}(...)`.
- Avoid import-time execution, local file reads, or current-working-directory assumptions so the final file can be copied after the experiment.
<!-- endsection -->

<!-- section:legacy_portability -->
Post-hoc portability constraint:
- `candidate.py` remains the H20 submission and `candidate(...)` remains the H20 entrypoint.
- When practical, put the optimized implementation behind a top-level helper named `{impl_name}_impl(...)`, then have `candidate(...)` call that helper with the declared H20 arguments.
- Keep helper kernels and helper functions at module top level rather than nested inside `candidate(...)`.
- Avoid import-time execution, local file reads, or current-working-directory assumptions so the final file can be copied or wrapped after the experiment.
- Do not add another speculative entrypoint.
- Do not replace the H20 interface with a differently named output file.
<!-- endsection -->

<!-- section:a0_retrieval -->
Retrieval protocol:
- This is the no-retrieval baseline.
- Do not call `tools/retrieve_context.py` or read `kb/`, `sources/raw_corpus/`, or `sources/raw_archive/`.
- Use only `task.json`, `candidate.py`, `run.sh`, harness output files, and your own reasoning.
<!-- endsection -->

<!-- section:a1_retrieval -->
Retrieval protocol:
- This group may use only raw-corpus vector RAG.
- Retrieval is mandatory at phase entry: call retrieval with `--group A1_raw_corpus_vector_rag` before acting in each selected phase.
- You must replace the placeholder after `--query` with your own specific query for this phase/problem. Do not use a generic phase-only query.
- Do not skip retrieval just because you already have an implementation or tuning idea.
- Do not read `kb/` directly and do not call retrieval with A2 or A3 groups.
- Save retrieved packets under `context_packets/`.
- Phase-specific retrieval commands from this workspace:
{phase_commands}
<!-- endsection -->

<!-- section:a2_retrieval -->
Retrieval protocol:
- This group may use only vector RAG over flattened KB units.
- Retrieval is mandatory at phase entry: call retrieval with `--group A2_kb_vector_rag` before acting in each selected phase.
- You must replace the placeholder after `--query` with your own specific query for this phase/problem. Do not use a generic phase-only query.
- Do not skip retrieval just because you already have an implementation or tuning idea.
- Do not read `sources/raw_corpus/` or `sources/raw_archive/` directly and do not call retrieval with A1 or A3 groups.
- Treat retrieved KB units as ordinary text snippets, not as structured ECC capsules.
- Save retrieved packets under `context_packets/`.
- Phase-specific retrieval commands from this workspace:
{phase_commands}
<!-- endsection -->

<!-- section:a3_retrieval -->
Retrieval protocol:
- This group uses ECC-KB structured context packets.
- Retrieval is mandatory at ECC phase entry: call retrieval with `--group A3_ecc_kb` before acting in each selected phase.
- You must replace the placeholder after `--query` with your own specific query for this phase/problem. Do not use a generic phase-only query.
- Do not skip retrieval just because you already have an implementation or tuning idea.
- Use the structured packet fields such as `must_obey`, `recommended_actions`, `anti_actions`, `validation_plan`, `stop_conditions`, and `evidence_refs`.
- Do not call retrieval with A1 or A2 groups, and do not browse raw corpus files directly.
- Save retrieved packets under `context_packets/`.
- Phase-specific commands from this workspace:
{phase_commands}
<!-- endsection -->

<!-- section:retrieval_outputs -->
Outputs:
- Final implementation: `candidate.py`.
- Required retrieved context evidence: `context_packets/*.json` for every phase you enter.
- Measurement outputs from `./run.sh`: `results.json`, `compile.log`, `correctness.log`, `benchmark.json`, `profile_summary.json`.
<!-- endsection -->

<!-- section:a0_outputs -->
Outputs:
- Final implementation: `candidate.py`.
- No retrieval output is expected for this no-retrieval baseline.
- Measurement outputs from `./run.sh`: `results.json`, `compile.log`, `correctness.log`, `benchmark.json`, `profile_summary.json`.
<!-- endsection -->
