#!/usr/bin/env python3
"""Fixed H20 MVP harness for Triton/PyTorch candidate kernels.

Candidate interface:
- reduction: candidate(x, reduce_op) -> y
- softmax: candidate(x) -> y
- layernorm: candidate(x, gamma, beta, eps) -> y
- matmul: candidate(a, b) -> c
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, sha256_file, short_hash, utc_now, write_json  # noqa: E402


def torch_dtype(name: str):
    import torch
    return {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[name]


def tolerance(dtype: str, op_family: str) -> tuple[float, float]:
    if dtype in ("fp32", "float32"):
        return 1e-4, 1e-4
    if dtype in ("bf16", "bfloat16"):
        return 8e-2, 8e-2
    if op_family in ("reduction", "layernorm"):
        return 5e-2, 5e-2
    return 3e-2, 3e-2


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("ecc_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ecc_candidate"] = module
    spec.loader.exec_module(module)
    for name in ("candidate", "run", "forward"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError("candidate.py must define callable candidate(), run(), or forward()")


def make_tensor_2d(shape: list[int], dtype_name: str, seed: int, layout: str, device: str):
    import torch
    torch.manual_seed(seed)
    b, n = int(shape[0]), int(shape[1])
    dtype = torch_dtype(dtype_name)
    if layout == "non_contiguous":
        return torch.randn((n, b), device=device, dtype=dtype).t()
    return torch.randn((b, n), device=device, dtype=dtype)


def make_inputs(task: dict[str, Any], test: dict[str, Any], base_seed: int, device: str):
    import torch
    op_family = task["op_family"]
    variant = task.get("variant", task.get("op_name", ""))
    dtype_name = test.get("dtype", task.get("dtype", "fp16"))
    layout = test.get("layout", "contiguous")
    seed = base_seed + int(test.get("seed_offset", 0))
    shape = test.get("shape", task["shape"])
    if op_family in {"reduction", "softmax", "layernorm"}:
        x = make_tensor_2d(shape, dtype_name, seed, layout, device)
        if op_family == "layernorm":
            n = int(shape[1])
            torch.manual_seed(seed + 17)
            gamma = torch.randn((n,), device=device, dtype=torch_dtype(dtype_name))
            beta = torch.randn((n,), device=device, dtype=torch_dtype(dtype_name))
            return x, gamma, beta, float(task.get("eps", 1e-5))
        if op_family == "reduction":
            return x, variant
        return (x,)
    if op_family == "matmul":
        dtype = torch_dtype(dtype_name)
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        torch.manual_seed(seed)
        a = torch.randn((m, k), device=device, dtype=dtype)
        torch.manual_seed(seed + 1)
        b = torch.randn((k, n), device=device, dtype=dtype)
        return a, b
    raise ValueError(f"unsupported op_family: {op_family}")


def reference(task: dict[str, Any], inputs: tuple[Any, ...]):
    import torch
    op_family = task["op_family"]
    if op_family == "reduction":
        x, reduce_op = inputs
        if reduce_op == "sum":
            return x.float().sum(dim=-1)
        if reduce_op == "max":
            return x.max(dim=-1).values
        raise ValueError(f"unsupported reduction op: {reduce_op}")
    if op_family == "softmax":
        (x,) = inputs
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
    if op_family == "layernorm":
        x, gamma, beta, eps = inputs
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        y = (xf - mean) / torch.sqrt(var + eps)
        return (y * gamma.float() + beta.float()).to(x.dtype)
    if op_family == "matmul":
        a, b = inputs
        return torch.matmul(a, b)
    raise ValueError(f"unsupported op_family: {op_family}")


def compare_outputs(actual: Any, expected: Any, dtype_name: str, op_family: str) -> dict[str, Any]:
    import torch
    if not torch.is_tensor(actual):
        return {"status": "fail", "reason": "candidate did not return a torch.Tensor"}
    if actual.shape != expected.shape:
        return {"status": "fail", "reason": f"shape mismatch actual={tuple(actual.shape)} expected={tuple(expected.shape)}"}
    atol, rtol = tolerance(dtype_name, op_family)
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=True)
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return {"status": "pass" if bool(ok) else "fail", "atol": atol, "rtol": rtol, "max_abs_error": max_abs, "max_rel_error": max_rel}


def run_suite(task: dict[str, Any], hidden: dict[str, Any], fn: Callable[..., Any], base_seed: int, device: str, log_lines: list[str]) -> dict[str, Any]:
    import torch
    suites = {"smoke": task.get("public_tests", [])[:1], "quick": task.get("public_tests", []), "hidden": hidden.get("hidden_tests", [])}
    result: dict[str, Any] = {}
    for suite_name, tests in suites.items():
        suite_records = []
        suite_ok = True
        for idx, test in enumerate(tests):
            try:
                inputs = make_inputs(task, test, base_seed + idx * 1000, device)
                expected = reference(task, inputs)
                actual = fn(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                cmp = compare_outputs(actual, expected, test.get("dtype", task.get("dtype", "fp16")), task["op_family"])
                record = {"test": test, **cmp}
                if cmp["status"] != "pass":
                    suite_ok = False
            except Exception as exc:
                suite_ok = False
                record = {"test": test, "status": "fail", "reason": str(exc), "traceback": traceback.format_exc(limit=8)}
            suite_records.append(record)
            if not suite_ok and suite_name == "smoke":
                break
        result[suite_name] = {"status": "pass" if suite_ok else "fail", "tests": suite_records}
        log_lines.append(f"{suite_name}: {result[suite_name]['status']}")
        if not suite_ok:
            break
    return result


def time_callable(fn: Callable[[], Any], warmup: int, repeats: int, device: str) -> dict[str, Any]:
    import torch
    samples: list[float] = []
    if device == "cuda":
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)))
    else:
        for _ in range(warmup):
            fn()
        for _ in range(repeats):
            start = time.perf_counter(); fn(); samples.append((time.perf_counter() - start) * 1000.0)
    samples = sorted(samples)
    p50 = statistics.median(samples) if samples else math.nan
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))] if samples else math.nan
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": statistics.mean(samples) if samples else math.nan, "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0, "min_ms": min(samples) if samples else math.nan, "samples": samples}


def benchmark(task: dict[str, Any], fn: Callable[..., Any], seed: int, device: str, warmup: int, repeats: int) -> dict[str, Any]:
    import torch
    test = task.get("public_tests", [])[0]
    inputs = make_inputs(task, test, seed, device)
    candidate_t = time_callable(lambda: fn(*inputs), warmup, repeats, device)
    eager_t = time_callable(lambda: reference(task, inputs), warmup, repeats, device)
    compile_t: dict[str, Any] = {"status": "not_run"}
    try:
        compiled_ref = torch.compile(lambda *args: reference(task, args), fullgraph=False)
        compile_t = {"status": "pass", **time_callable(lambda: compiled_ref(*inputs), max(5, warmup // 5), repeats, device)}
    except Exception as exc:
        compile_t = {"status": "fail", "reason": str(exc)}
    cand_p50 = float(candidate_t["p50_ms"])
    eager_p50 = float(eager_t["p50_ms"])
    compile_p50 = float(compile_t.get("p50_ms", math.nan)) if compile_t.get("status") == "pass" else math.nan
    return {"measurement": {"warmup": warmup, "repeats": repeats}, "candidate": candidate_t, "eager": eager_t, "torch_compile": compile_t, "speedup_vs_eager_p50": eager_p50 / cand_p50 if cand_p50 > 0 else 0.0, "speedup_vs_torch_compile_p50": compile_p50 / cand_p50 if cand_p50 > 0 and not math.isnan(compile_p50) else 0.0}


def profile_summary(task: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    symptom = "compute_bound" if task.get("op_family") == "matmul" else "memory_bound"
    speedup = bench.get("speedup_vs_eager_p50", 0.0) if bench else 0.0
    actions = ["verify_p95_stability", "try_small_knob_search_only"] if speedup >= 1.0 else ["inspect_memory_access_pattern", "reduce_register_pressure", "retune_tile_or_block_size"]
    return {"dominant_symptom": symptom, "evidence": {"heuristic": "op-family fallback summary; replace with NCU/Triton profiler when available", "speedup_vs_eager_p50": speedup}, "candidate_actions": actions}


def status_at(data: dict[str, Any], path: str, default: str = "not_run") -> str:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return str(cur)


def print_summary(results: dict[str, Any], out_dir: Path) -> None:
    print("H20 harness summary")
    print(f"compile: {status_at(results, 'compile.status')}")
    print(
        "correctness: "
        f"smoke={status_at(results, 'correctness.smoke.status')} "
        f"quick={status_at(results, 'correctness.quick.status')} "
        f"hidden={status_at(results, 'correctness.hidden.status')}"
    )
    benchmark = results.get("benchmark", {})
    if benchmark:
        print(
            "performance: "
            f"p50_ms={float(benchmark.get('latency_p50_ms', math.nan)):.6g} "
            f"p95_ms={float(benchmark.get('latency_p95_ms', math.nan)):.6g} "
            f"speedup_vs_eager={float(benchmark.get('speedup_vs_eager_p50', 0.0)):.4g} "
            f"speedup_vs_torch_compile={float(benchmark.get('speedup_vs_torch_compile_p50', 0.0)):.4g}"
        )
    else:
        print("performance: not_run")
    diagnosis = results.get("diagnosis") or status_at(results, "compile.reason", "")
    if diagnosis:
        print(f"diagnosis: {diagnosis}")
    print("outputs:")
    for name in ("results.json", "compile.log", "correctness.log", "benchmark.json", "profile_summary.json"):
        print(f"  {name}: {out_dir / name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-tests", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true", help="Fail instead of silently falling back to CPU when CUDA is unavailable.")
    args = parser.parse_args()

    import torch
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    task = read_json(args.task)
    hidden = read_json(args.hidden_tests, default={"hidden_tests": []}) if args.hidden_tests else {"hidden_tests": []}
    candidate_path = Path(args.candidate)
    warmup = int(args.warmup if args.warmup is not None else task.get("budget", {}).get("warmup", 100))
    repeats = int(args.repeats if args.repeats is not None else task.get("budget", {}).get("repeats", 500))
    requested_device = args.device
    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available and args.require_cuda:
        results = {
            "schema_version": "0.1",
            "run_id": f"harness_{short_hash([task, args.seed, sha256_file(candidate_path), 'cuda_unavailable'])}",
            "timestamp": utc_now(),
            "task_id": task.get("task_id"),
            "op_family": task.get("op_family"),
            "candidate_hash": sha256_file(candidate_path),
            "compile": {"status": "fail", "reason": "CUDA requested but torch.cuda.is_available() is false"},
            "anti_cheating": {"status": "not_run", "judge": "llm_judge_pending", "issues": []},
            "correctness": {},
            "benchmark": {},
            "profile_summary": {},
            "cost": {"iterations": 1, "gpu_benchmark_runs": 0, "wall_time_s": 0},
            "runtime": {
                "requested_device": requested_device,
                "actual_device": "none",
                "cuda_available": False,
                "cuda_device_count": int(torch.cuda.device_count()),
            },
            "final_decision": "FAIL",
            "diagnosis": "CUDA requested but unavailable; H20 benchmark was not run.",
        }
        write_json(out_dir / "results.json", results)
        (out_dir / "compile.log").write_text(results["diagnosis"] + "\n", encoding="utf-8")
        (out_dir / "correctness.log").write_text("", encoding="utf-8")
        write_json(out_dir / "benchmark.json", {})
        write_json(out_dir / "profile_summary.json", {})
        print_summary(results, out_dir)
        return 1
    device = requested_device if requested_device != "cuda" or cuda_available else "cpu"
    started = time.time()
    compile_log: list[str] = []
    correctness_log: list[str] = []
    results: dict[str, Any] = {"schema_version": "0.1", "run_id": f"harness_{short_hash([task, args.seed, sha256_file(candidate_path)])}", "timestamp": utc_now(), "task_id": task.get("task_id"), "op_family": task.get("op_family"), "candidate_hash": sha256_file(candidate_path), "compile": {"status": "not_run"}, "anti_cheating": {"status": "not_run", "judge": "llm_judge_pending", "issues": []}, "correctness": {}, "benchmark": {}, "profile_summary": {}, "cost": {"iterations": 1, "gpu_benchmark_runs": 0, "wall_time_s": 0}, "runtime": {"requested_device": requested_device, "actual_device": device, "cuda_available": bool(cuda_available), "cuda_device_count": int(torch.cuda.device_count())}}
    try:
        fn = load_candidate(candidate_path)
        results["compile"] = {"status": "pass"}; compile_log.append("candidate import: pass")
    except Exception as exc:
        results["compile"] = {"status": "fail", "reason": str(exc), "traceback": traceback.format_exc(limit=12)}
        results["final_decision"] = "FAIL"; results["cost"]["wall_time_s"] = time.time() - started
        write_json(out_dir / "results.json", results); (out_dir / "compile.log").write_text(traceback.format_exc(), encoding="utf-8")
        (out_dir / "correctness.log").write_text("", encoding="utf-8"); write_json(out_dir / "benchmark.json", {}); write_json(out_dir / "profile_summary.json", {})
        print_summary(results, out_dir); return 1

    correctness = run_suite(task, hidden, fn, args.seed, device, correctness_log)
    results["correctness"] = correctness
    hidden_ok = str(correctness.get("hidden", {}).get("status", "fail")) == "pass"
    bench: dict[str, Any] = {}; prof: dict[str, Any] = {}
    if hidden_ok:
        bench = benchmark(task, fn, args.seed, device, warmup, repeats)
        prof = profile_summary(task, bench)
        results["benchmark"] = {"latency_p50_ms": bench["candidate"]["p50_ms"], "latency_p95_ms": bench["candidate"]["p95_ms"], "latency_mean_ms": bench["candidate"]["mean_ms"], "latency_std_ms": bench["candidate"]["std_ms"], "speedup_vs_eager_p50": bench.get("speedup_vs_eager_p50", 0.0), "speedup_vs_torch_compile_p50": bench.get("speedup_vs_torch_compile_p50", 0.0)}
        results["profile_summary"] = prof; results["cost"]["gpu_benchmark_runs"] = repeats
        results["final_decision"] = "KEEP" if results["benchmark"]["speedup_vs_eager_p50"] >= 1.0 else "DISCARD"
    else:
        results["final_decision"] = "FAIL"
        results["diagnosis"] = "hidden correctness failed"
    results["cost"]["wall_time_s"] = time.time() - started
    write_json(out_dir / "results.json", results); write_json(out_dir / "benchmark.json", bench); write_json(out_dir / "profile_summary.json", prof)
    (out_dir / "compile.log").write_text("\n".join(compile_log) + "\n", encoding="utf-8")
    (out_dir / "correctness.log").write_text("\n".join(correctness_log) + "\n", encoding="utf-8")
    print_summary(results, out_dir)
    return 0 if results["final_decision"] in {"KEEP", "DISCARD"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
