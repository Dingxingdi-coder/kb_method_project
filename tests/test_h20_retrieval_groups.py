import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = REPO_ROOT / "artifacts" / "h20" / "tasks_round1" / "round1_softmax_row_softmax_4096x1024_bf16.json"
BACKEND = REPO_ROOT / "artifacts" / "h20" / "backend_h20_v0.json"
MODEL = Path("/data/dxd/models/bge-small-en-v1.5")


class H20RetrievalGroupSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.raw_index = Path(cls._tmpdir.name) / "raw_index"
        cls.kb_index = Path(cls._tmpdir.name) / "kb_index"
        if not MODEL.exists():
            cls.vector_skip = f"missing embedding model: {MODEL}"
            return
        cls.vector_skip = ""
        for kind, out in [("raw_corpus", cls.raw_index), ("kb", cls.kb_index)]:
            cmd = [
                sys.executable,
                "tools/build_vector_index.py",
                "--kind",
                kind,
                "--embedding-model-path",
                str(MODEL),
                "--output",
                str(out),
                "--repo-root",
                str(REPO_ROOT),
                "--source-root",
                str(REPO_ROOT / "sources"),
                "--kb-root",
                str(REPO_ROOT / "kb"),
                "--kb-version",
                "v0",
                "--batch-size",
                "64",
            ]
            try:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except subprocess.CalledProcessError as exc:
                cls.vector_skip = exc.stderr or exc.stdout or str(exc)
                return

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir"):
            cls._tmpdir.cleanup()

    def retrieve(self, group: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "context.json"
            cmd = [
                sys.executable,
                "tools/retrieve_context.py",
                "--task",
                str(TASK),
                "--backend",
                str(BACKEND),
                "--group",
                group,
                "--phase",
                "performance_optimize",
                "--kb-version",
                "v0",
                "--raw-vector-index",
                str(self.raw_index),
                "--kb-vector-index",
                str(self.kb_index),
                "--query",
                "bf16 row softmax 4096x1024 H20 performance optimize masked stable triton num_warps",
                "--out",
                str(out),
            ]
            if group != "A3_ecc_kb" and not self.vector_skip:
                cmd.extend(["--embedding-model-path", str(MODEL)])
            if group == "A3_ecc_kb":
                if self.vector_skip:
                    self.skipTest(self.vector_skip)
                cmd.extend(["--embedding-model-path", str(MODEL)])
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return json.loads(out.read_text(encoding="utf-8"))

    def test_a0_has_no_retrieval(self) -> None:
        context = self.retrieve("A0_prompt")
        self.assertEqual(context["group"], "A0_prompt")
        self.assertEqual(context["retrieval_mode"], "none")
        self.assertEqual(context["retrieved_item_ids"], [])

    def test_a1_reads_frozen_raw_corpus_only(self) -> None:
        if self.vector_skip:
            self.skipTest(self.vector_skip)
        context = self.retrieve("A1_raw_corpus_vector_rag")
        self.assertEqual(context["group"], "A1_raw_corpus_vector_rag")
        self.assertEqual(context["retrieval_mode"], "vector_rag")
        self.assertEqual(context["query_source"], "agent_query")
        self.assertIn("bf16 row softmax", context["retrieval_query"])
        self.assertEqual(context["source_corpus_version"], "raw_corpus_v0")
        self.assertEqual(context["embedding_model_id"], "BAAI/bge-small-en-v1.5")
        self.assertTrue(context["vector_index_hash"])
        self.assertTrue(context["document_chunks"])
        for item in context["document_chunks"]:
            self.assertIn(item["source_type"], {"raw_corpus", "raw_archive"})
            self.assertTrue(item["source_path"].startswith(("sources/raw_corpus/", "sources/raw_archive/")))

    def test_a2_reads_kb_units_without_static_rulebook(self) -> None:
        if self.vector_skip:
            self.skipTest(self.vector_skip)
        context = self.retrieve("A2_kb_vector_rag")
        self.assertEqual(context["group"], "A2_kb_vector_rag")
        self.assertEqual(context["retrieval_mode"], "vector_rag")
        self.assertEqual(context["query_source"], "agent_query")
        self.assertIn("bf16 row softmax", context["retrieval_query"])
        self.assertEqual(context["embedding_model_id"], "BAAI/bge-small-en-v1.5")
        self.assertTrue(context["vector_index_hash"])
        self.assertTrue(context["kb_units"])
        self.assertNotIn("rule_tail_mask", json.dumps(context, ensure_ascii=False))
        for item in context["kb_units"]:
            self.assertEqual(item["source_type"], "kb_unit")
            self.assertTrue(item["source_path"].startswith("kb/"))

    def test_a3_uses_structured_ecc_packet(self) -> None:
        context = self.retrieve("A3_ecc_kb")
        self.assertEqual(context["group"], "A3_ecc_kb")
        self.assertEqual(context["retrieval_mode"], "phase_and_evidence_aware_context_packet")
        self.assertEqual(context["query_source"], "agent_query_plus_task_phase_backend_filters")
        self.assertIn("bf16 row softmax", context["retrieval_query"])
        self.assertEqual(context["hard_filters"]["phase"], "performance_optimize")
        self.assertEqual(context["hard_filters"]["operator_family"], "softmax")
        self.assertEqual(context["backend_filters"]["backend"], "nvidia-triton-cuda")
        self.assertEqual(context["semantic_retrieval"]["stage"], "after_hard_filter")
        self.assertTrue(context["semantic_retrieval"]["applied"], context["semantic_retrieval"].get("error"))
        self.assertIn("evidence_strength", context["rerank_policy"])
        self.assertTrue(context["retrieved_capsules"])
        self.assertEqual(context["retrieved_item_ids"], context["retrieved_capsules"])
        for item in context["retrieved_items"]:
            self.assertIn("score", item)
            self.assertIn("semantic_cosine", item["score"])

    def test_legacy_names_are_canonicalized(self) -> None:
        if self.vector_skip:
            self.skipTest(self.vector_skip)
        context = self.retrieve("A2_rulebook")
        self.assertEqual(context["group"], "A2_kb_vector_rag")
        self.assertTrue(context["kb_units"])

    def test_agent_prompts_define_group_retrieval_boundaries(self) -> None:
        expected = {
            "A0_prompt": ["no-retrieval baseline", "Do not call `tools/retrieve_context.py`"],
            "A1_raw_corpus_vector_rag": [
                "--group A1_raw_corpus_vector_rag",
                "--kb-root",
                "--source-root",
                "--raw-vector-index",
                "Do not read `kb/` directly",
                "Retrieval is mandatory at phase entry",
                "Required retrieved context evidence",
            ],
            "A2_kb_vector_rag": [
                "--group A2_kb_vector_rag",
                "--kb-root",
                "--source-root",
                "--kb-vector-index",
                "vector RAG over flattened KB units",
                "Retrieval is mandatory at phase entry",
                "Required retrieved context evidence",
            ],
            "A3_ecc_kb": [
                "--group A3_ecc_kb",
                "--kb-root",
                "--source-root",
                "structured packet fields",
                "Retrieval is mandatory at ECC phase entry",
                "Required retrieved context evidence",
            ],
        }
        common_snippets = [
            "Edit `candidate.py` for this ECC-KB experiment",
            "Read `task.json` first",
            "Do not read hidden test files or modify the harness",
            "`correctness_repair`",
            "`performance_optimize`",
            "`autotune`",
            "`./run.sh --stage compile`",
            "`./run.sh --stage benchmark`",
            "Use staged harness runs",
            "Post-hoc portability constraint",
            "`candidate.py` remains the H20 submission",
            "Do not add another speculative entrypoint",
            "Do not replace the H20 interface with a differently named output file",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for group, snippets in expected.items():
                out = Path(tmp) / group
                cmd = [
                    sys.executable,
                    "experiments/h20/run_agent_session.py",
                    "--group",
                    group,
                    "--task",
                    str(TASK),
                    "--backend",
                    str(BACKEND),
                    "--phase",
                    "generate",
                    "--kb-version",
                    "v0",
                "--out",
                str(out),
                "--embedding-model-path",
                str(MODEL),
            ]
                subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                prompt = (out / "agent_prompt.md").read_text(encoding="utf-8")
                self.assertTrue((out / "context_packets").is_dir())
                self.assertFalse((out / "context_packet.json").exists())
                self.assertFalse((out / "notes.md").exists())
                task = json.loads((out / "task.json").read_text(encoding="utf-8"))
                self.assertEqual(task["agent_allowed_files"], ["candidate.py", "context_packets/"])
                for snippet in snippets + common_snippets:
                    self.assertIn(snippet, prompt)

    def test_kernelbenchx_overlap_prompt_for_expanded_pilot_softmax(self) -> None:
        task = REPO_ROOT / "artifacts" / "h20" / "tasks_expanded_pilot" / "expanded_pilot_softmax_softmax.json"
        if not task.exists():
            self.skipTest(f"missing expanded pilot task: {task}")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a3_softmax"
            cmd = [
                sys.executable,
                "experiments/h20/run_agent_session.py",
                "--group",
                "A3_ecc_kb",
                "--task",
                str(task),
                "--backend",
                str(BACKEND),
                "--phase",
                "generate",
                "--kb-version",
                "v0",
                "--out",
                str(out),
                "--embedding-model-path",
                str(MODEL),
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            prompt = (out / "agent_prompt.md").read_text(encoding="utf-8")
            candidate = (out / "candidate.py").read_text(encoding="utf-8")
            self.assertIn("This task requires the top-level callable `def softmax(input, dim, dtype=None)`", prompt)
            self.assertIn("Submission interface", prompt)
            self.assertIn("Follow the public callable semantics in `task.json`", prompt)
            self.assertIn("The function name must remain `softmax`", prompt)
            self.assertIn("Do not add or rely on an extra `candidate(...)` wrapper for this task", prompt)
            self.assertNotIn("KernelBench-X task", prompt)
            self.assertNotIn("candidate(...) remains the H20 entrypoint", prompt)
            self.assertIn("def softmax(input, dim, dtype=None):", candidate)
            self.assertNotIn("def candidate", candidate)

    def test_kernelbenchx_overlap_prompt_for_expanded_pilot_layout(self) -> None:
        task = REPO_ROOT / "artifacts" / "h20" / "tasks_expanded_pilot" / "expanded_pilot_layout_index_select.json"
        if not task.exists():
            self.skipTest(f"missing expanded pilot task: {task}")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a3_index_select"
            cmd = [
                sys.executable,
                "experiments/h20/run_agent_session.py",
                "--group",
                "A3_ecc_kb",
                "--task",
                str(task),
                "--backend",
                str(BACKEND),
                "--phase",
                "generate",
                "--kb-version",
                "v0",
                "--out",
                str(out),
                "--embedding-model-path",
                str(MODEL),
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            prompt = (out / "agent_prompt.md").read_text(encoding="utf-8")
            candidate = (out / "candidate.py").read_text(encoding="utf-8")
            self.assertIn("This task requires the top-level callable `def index_select(input, dim, index)`", prompt)
            self.assertIn("Submission interface", prompt)
            self.assertIn("Follow the public callable semantics in `task.json`", prompt)
            self.assertIn("Do not add or rely on an extra `candidate(...)` wrapper for this task", prompt)
            self.assertNotIn("KernelBench-X task", prompt)
            self.assertNotIn("candidate(...) remains the H20 entrypoint", prompt)
            self.assertIn("def index_select(input, dim, index):", candidate)
            self.assertNotIn("def candidate", candidate)

    def test_budget_audit_is_posthoc_and_marks_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "A3_ecc_kb" / "task" / "seed0"
            metrics = ws / ".codex_h20_metrics"
            metrics.mkdir(parents=True)
            (ws / "task.json").write_text(
                json.dumps({"task_id": "demo_task", "op_family": "pointwise"}),
                encoding="utf-8",
            )
            candidate = ws / "candidate.py"
            candidate.write_text("def candidate(x):\n    return x\n", encoding="utf-8")
            (metrics / "summary.json").write_text(
                json.dumps(
                    {
                        "meta": {"group": "A3_ecc_kb"},
                        "candidate_count": 3,
                        "totals": {
                            "compile_attempts": 5,
                            "correctness_runs": 4,
                            "benchmark_runs": 2,
                            "harness_runs": 7,
                        },
                        "agents": {"agent0": {"wall_time_s": 12.5}},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "audit.json"
            cmd = [
                sys.executable,
                "experiments/h20/audit_budgets.py",
                "--runs",
                str(root),
                "--out",
                str(out),
                "--max-agent-wall-time-s",
                "10",
                "--max-candidates",
                "2",
                "--max-compile-attempts",
                "5",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "post_hoc_observational_only")
            self.assertEqual(report["workspace_count"], 1)
            self.assertEqual(report["over_budget_count"], 1)
            row = report["runs"][0]
            self.assertEqual(row["group"], "A3_ecc_kb")
            self.assertTrue(row["over_budget"])
            self.assertIn("agent_wall_time_s", row["over_budget_metrics"])
            self.assertIn("candidate_count", row["over_budget_metrics"])
            self.assertEqual(candidate.read_text(encoding="utf-8"), "def candidate(x):\n    return x\n")


if __name__ == "__main__":
    unittest.main()
