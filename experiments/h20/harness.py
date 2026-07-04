#!/usr/bin/env python3
"""Fixed H20 harness for candidate Triton/PyTorch kernels.

Candidate contract:
- `candidate.py` may define `run(*inputs, task=None)`.
- For compatibility, it may define op-specific functions: `row_softmax(x)`, `row_sum(x)`,
  `row_max(x)`, `layernorm_forward(x, gamma, beta, eps)`, or `matmul(a, b)`.
The harness owns reference generation, correctness gates, timing, and output JSON files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def torch_dtype(name: str):
    import torch

    return {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[name]


def tolerances(dtype: str) -> tuple[float, float]:
    if dtype in {"fp32", "float32"}:
        return 1e-4, 1e-4
    if dtype in {"bf16", "bfloat16"}:
        return 4e-2, 4e-2
    return 2e-2, 2e-2


def import_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(task: dict[str, Any], seed: int, non_contiguous: bool = False):
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    dtype = torch_dtype(task["dtype"])
    op = task["op_family"]
    device = "cuda"
    if op in {"softmax", "reduction", "layernorm"}:
        shape = list(task["shape"])
        x = torch.randn(*shape, device=device, dtype=dtype)
        if op == "softmax":
            x = x * 3.0
            if shape[-1] > 1:
                x[..., 0] = 50
                x[..., -1] = -50
        if non_contiguous and len(shape) == 2:
            # Preserve logical shape while changing stride.
            x = torch.randn(shape[1], shape[0], device=device, dtype=dtype).t()
        if op == "layernorm":
            gamma = torch.randn(shape[-1], device=device, dtype=dtype)
            beta = torch.randn(shape[-1], device=device, dtype=dtype)
            return [x, gamma, beta]
        return [x]
    if op == "matmul":
        m, n, k = task["shape"]
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        return [a, b]
    raise ValueError(f"Unsupported op_family: {op}")


def reference(task: dict[str, Any], inputs: list[Any]):
    import torch

    op = task["op_family"]
    if op == "softmax":
        return torch.softmax(inputs[0], dim=-1)
    if op == "reduction":
        if task["op_name"] == "row_max":
            return torch.max(inputs[0], dim=-1).values
        return torch.sum(inputs[0].to(torch.float32), dim=-1).to(inputs[0].dtype)
    if op == "layernorm":
        x, gamma, beta = inputs
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps=float(task.get("eps", 1e-5)))
    if op == "matmul":
        return inputs[0] @ inputs[1]
    raise ValueError(f"Unsupported op_family: {op}")


def candidate_callable(module: Any, task: dict[str, Any]) -> Callable[..., Any]:
    if hasattr(module, "run"):
        return lambda *inputs: module.run(*inputs, task=task)
    name_map = {
        "softmax": "row_softmax",
        "layernorm": "layernorm_forward",
        "matmul": "matmul",
    }
    if task["op_family"] == "reduction":
        fn_name = "row_max" if task["op_name"] == "row_max" else "row_sum"
    else:
        fn_name = name_map[task["op_family"]]
    if not hasattr(module, fn_name):
        raise AttributeError(f"candidate.py must define run(...) or {fn_name}(...)")
    fn = getattr(module, fn_name)
    if task["op_family"] == "layernorm":
        return lambda x, gamma, beta: fn(x, gamma, beta, float(task.get("eps", 1e-5)))
    return fn


def compare_outputs(task: dict[str, Any], got: Any, expected: Any) -> dict[str, Any]:
    import torch

    if isinstance(got, (tuple, list)):
        got = got[0]
    atol, rtol = tolerances(task["dtype"])
    ok = bool(torch.allclose(got, expected, atol=atol, rtol=rtol, equal_nan=True))
    max_abs = float((got - expected).abs().max().item()) if got.numel() else 0.0
    denom = expected.abs().clamp_min(1e-12)
    max_rel = float(((got - expected).abs() / denom).max().item()) if got.numel() else 0.0
    return {"status": "pass" if ok else "fail", "atol": atol, "rtol": rtol, "max_abs_error": max_abs, "max_rel_error": max_rel}


def run_gate(task: dict[str, Any], fn: Callable[..., Any], seeds: list[int], non_contiguous: bool = False) -> dict[str, Any]:
    import torch

    cases = []
    for seed in seeds:
        inputs = make_inputs(task, seed, non_contiguous=non_contiguous)
        expected = reference(task, inputs)
        try:
            got = fn(*inputs)
            torch.cuda.synchronize()
            result = compare_outputs(task, got, expected)
        except Exception as exc:
            result = {"status": "fail", "exception": repr(exc), "traceback": traceback.format_exc(limit=5)}
        result.update({"seed": seed, "non_contiguous": non_contiguous})
        cases.append(result)
        if result["status"] != "pass":
            break
    return {"status": "pass" if all(c["status"] == "pass" for c in cases) else "fail", "cases": cases}


def benchmark(task: dict[str, Any], fn: Callable[..., Any], warmup: int, repeats: int, seed: int) -> dict[str, Any]:
    import numpy as np
    import torch

    inputs = make_inputs(task, seed, non_contiguous=False)
    # Candidate timing.
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))

    # Eager reference timing.
    for _ in range(max(10, warmup // 10)):
        reference(task, inputs)
    torch.cuda.synchronize()
    eager = []
    for _ in range(max(30, min(repeats, 100))):
        start.record()
        reference(task, inputs)
        end.record()
        torch.cuda.synchronize()
        eager.append(float(start.elapsed_time(end)))

    arr = np.array(samples, dtype=float)
    eager_arr = np.array(eager, dtype=float)
    p50 = float(np.percentile(arr, 50))
    eager_p50 = float(np.percentile(eager_arr, 50))
    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
        "latency_min_ms": float(arr.min()),
        "eager_latency_p50_ms": eager_p50,
        "speedup_vs_eager_p50": eager_p50 / p50 if p50 > 0 else None,
        "warmup": warmup,
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = load_json(args.task)
    started = time.time()
    results: dict[str, Any] = {"run_id": out_dir.name, "task": task, "compile": {}, "correctness": {}, "benchmark": {}, "cost": {"iterations": 1}}

    try:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            print("WARN: CUDA_VISIBLE_DEVICES is not set; benchmark noise may increase.")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        module = import_candidate(Path(args.candidate))
        fn = candidate_callable(module, task)
        results["compile"] = {"status": "pass"}
    except Exception as exc:
        results["compile"] = {"status": "fail", "exception": repr(exc), "traceback": traceback.format_exc(limit=10)}
        write_json(out_dir / "results.json", results)
        Path(out_dir / "compile.log").write_text(results["compile"].get("traceback", ""), encoding="utf-8")
        raise SystemExit(1)

    results["correctness"]["smoke"] = run_gate(task, fn, [args.seed])
    if results["correctness"]["smoke"]["status"] == "pass":
        results["correctness"]["quick"] = run_gate(task, fn, [args.seed + 1, args.seed + 2])
    if results["correctness"].get("quick", {}).get("status") == "pass":
        results["correctness"]["hidden"] = run_gate(task, fn, [1009 + args.seed, 2003 + args.seed, 3001 + args.seed])
    if results["correctness"].get("hidden", {}).get("status") == "pass" and task["op_family"] in {"softmax", "reduction", "layernorm"}:
        results["correctness"]["robustness"] = run_gate(task, fn, [4001 + args.seed], non_contiguous=True)

    hidden_pass = results["correctness"].get("hidden", {}).get("status") == "pass"
    if hidden_pass and not args.skip_benchmark:
        try:
            results["benchmark"] = benchmark(task, fn, args.warmup, args.repeats, args.seed)
            write_json(out_dir / "benchmark.json", results["benchmark"])
            write_json(out_dir / "profile_summary.json", {"dominant_symptom": "unknown", "evidence": [], "candidate_actions": []})
        except Exception as exc:
            results["benchmark"] = {"status": "fail", "exception": repr(exc), "traceback": traceback.format_exc(limit=10)}

    results["cost"]["wall_time_s"] = time.time() - started
    write_json(out_dir / "results.json", results)
    write_json(out_dir / "correctness.log.json", results["correctness"])
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not hidden_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
