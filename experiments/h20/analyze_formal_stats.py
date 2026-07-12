#!/usr/bin/env python3
"""Compute formal H20 protocol statistics from analyze_results.py JSON.

This script is intentionally post-hoc. It does not read or modify workspaces,
run candidates, enforce budgets, or judge legality. It consumes the merged JSON
produced by experiments/h20/analyze_results.py after official evaluator and
budget audit reports have been merged.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, write_json  # noqa: E402


A3_GROUP = "A3_ecc_kb"
BASELINES = ["A2_kb_vector_rag", "A1_raw_corpus_vector_rag", "A0_prompt"]
BUDGET_FIELDS = [
    "budget_candidate_count",
    "budget_compile_attempts",
    "budget_correctness_runs",
    "budget_benchmark_runs",
    "budget_harness_runs",
]


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite(value: Any) -> bool:
    return math.isfinite(as_float(value))


def median(values: list[Any]) -> float:
    nums = [as_float(v) for v in values]
    nums = [v for v in nums if math.isfinite(v)]
    return statistics.median(nums) if nums else math.nan


def mean(values: list[Any]) -> float:
    nums = [as_float(v) for v in values]
    nums = [v for v in nums if math.isfinite(v)]
    return sum(nums) / len(nums) if nums else math.nan


def pct(value: float) -> float:
    return value * 100.0 if math.isfinite(value) else math.nan


def rate(pass_count: int, total: int) -> float:
    return pass_count / total if total else math.nan


def wilson_interval(pass_count: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    phat = pass_count / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def mcnemar_exact_pvalue(a3_pass_base_fail: int, base_pass_a3_fail: int) -> float:
    n = a3_pass_base_fail + base_pass_a3_fail
    if n == 0:
        return math.nan
    lo = min(a3_pass_base_fail, base_pass_a3_fail)
    hi = max(a3_pass_base_fail, base_pass_a3_fail)
    lower = binomial_cdf(lo, n, 0.5)
    upper = 1.0 - binomial_cdf(hi - 1, n, 0.5)
    return min(1.0, 2.0 * min(lower, upper))


def block_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("task_id", "")), str(row.get("run", ""))


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("group", "unknown"))].append(row)
    return dict(out)


def rows_by_group_block(rows: list[dict[str, Any]]) -> dict[tuple[str, tuple[str, str]], dict[str, Any]]:
    out: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("group", "unknown")), block_key(row))] = row
    return out


def oracle_latency(row: dict[str, Any]) -> float:
    official = as_float(row.get("official_oracle_best_latency_p50_ms"))
    if math.isfinite(official):
        return official
    return math.nan


def final_correct(row: dict[str, Any]) -> int:
    official = row.get("official_final_hidden_pass")
    if official is not None:
        try:
            return 1 if int(official) == 1 else 0
        except (TypeError, ValueError):
            pass
    try:
        return 1 if int(row.get("hidden_pass", 0) or 0) == 1 else 0
    except (TypeError, ValueError):
        return 0


def paired_deltas(rows: list[dict[str, Any]], baseline: str, latency_getter=oracle_latency) -> list[dict[str, Any]]:
    by_key = rows_by_group_block(rows)
    blocks = sorted({block_key(row) for row in rows})
    out = []
    for key in blocks:
        a3 = by_key.get((A3_GROUP, key))
        base = by_key.get((baseline, key))
        if not a3 or not base:
            continue
        a3_lat = latency_getter(a3)
        base_lat = latency_getter(base)
        if not (math.isfinite(a3_lat) and math.isfinite(base_lat) and a3_lat > 0 and base_lat > 0):
            continue
        delta = math.log(a3_lat) - math.log(base_lat)
        out.append(
            {
                "task_id": key[0],
                "run": key[1],
                "baseline": baseline,
                "a3_latency_p50_ms": a3_lat,
                "baseline_latency_p50_ms": base_lat,
                "delta_log_latency": delta,
            }
        )
    return out


def summarize_deltas(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(d["delta_log_latency"]) for d in deltas]
    mean_delta = mean(values)
    median_delta = median(values)
    return {
        "paired_block_count": len(values),
        "mean_delta_log_latency": mean_delta,
        "median_delta_log_latency": median_delta,
        "mean_latency_change_pct": pct(math.exp(mean_delta) - 1.0) if math.isfinite(mean_delta) else math.nan,
        "median_latency_change_pct": pct(math.exp(median_delta) - 1.0) if math.isfinite(median_delta) else math.nan,
        "mean_latency_reduction_pct": pct(1.0 - math.exp(mean_delta)) if math.isfinite(mean_delta) else math.nan,
        "median_latency_reduction_pct": pct(1.0 - math.exp(median_delta)) if math.isfinite(median_delta) else math.nan,
    }


def bootstrap_ci_by_task(
    deltas: list[dict[str, Any]],
    iterations: int,
    bootstrap_run: int,
) -> dict[str, Any]:
    if not deltas:
        return {"iterations": iterations, "ci_low": math.nan, "ci_high": math.nan}
    by_task: dict[str, list[float]] = defaultdict(list)
    for item in deltas:
        by_task[str(item["task_id"])].append(float(item["delta_log_latency"]))
    tasks = sorted(by_task)
    rng = random.Random(bootstrap_run)
    samples: list[float] = []
    for _ in range(iterations):
        vals: list[float] = []
        for _task in tasks:
            sampled_task = rng.choice(tasks)
            vals.extend(by_task[sampled_task])
        if vals:
            samples.append(sum(vals) / len(vals))
    samples.sort()
    if not samples:
        return {"iterations": iterations, "ci_low": math.nan, "ci_high": math.nan}
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {
        "iterations": iterations,
        "ci_low_delta_log_latency": lo,
        "ci_high_delta_log_latency": hi,
        "ci_low_latency_change_pct": pct(math.exp(lo) - 1.0),
        "ci_high_latency_change_pct": pct(math.exp(hi) - 1.0),
    }


def primary_comparisons(rows: list[dict[str, Any]], bootstrap_iters: int, bootstrap_run: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, baseline in enumerate(BASELINES):
        deltas = paired_deltas(rows, baseline)
        out[baseline] = {
            **summarize_deltas(deltas),
            "bootstrap_task_95ci": bootstrap_ci_by_task(deltas, bootstrap_iters, bootstrap_run + idx),
        }
    return out


def correctness_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = group_rows(rows)
    summary: dict[str, Any] = {}
    for group, items in sorted(groups.items()):
        total = len(items)
        passes = sum(final_correct(row) for row in items)
        lo, hi = wilson_interval(passes, total)
        summary[group] = {
            "n": total,
            "pass_count": passes,
            "pass_rate": rate(passes, total),
            "wilson_95ci_low": lo,
            "wilson_95ci_high": hi,
        }

    paired: dict[str, Any] = {}
    by_key = rows_by_group_block(rows)
    blocks = sorted({block_key(row) for row in rows})
    for baseline in BASELINES:
        a3_pass_base_fail = 0
        base_pass_a3_fail = 0
        paired_count = 0
        for key in blocks:
            a3 = by_key.get((A3_GROUP, key))
            base = by_key.get((baseline, key))
            if not a3 or not base:
                continue
            paired_count += 1
            a3_ok = final_correct(a3)
            base_ok = final_correct(base)
            if a3_ok and not base_ok:
                a3_pass_base_fail += 1
            elif base_ok and not a3_ok:
                base_pass_a3_fail += 1
        paired[baseline] = {
            "paired_block_count": paired_count,
            "a3_pass_baseline_fail": a3_pass_base_fail,
            "baseline_pass_a3_fail": base_pass_a3_fail,
            "mcnemar_exact_pvalue": mcnemar_exact_pvalue(a3_pass_base_fail, base_pass_a3_fail),
        }
    return {"group_pass_rates": summary, "paired_mcnemar": paired}


def budget_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key_name in ("group", "op_family"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key_name, "unknown"))].append(row)
        out[key_name] = {}
        for name, items in sorted(buckets.items()):
            observed = [row for row in items if row.get("budget_over_budget") is not None]
            over = sum(1 for row in observed if int(row.get("budget_over_budget") or 0) == 1)
            out[key_name][name] = {
                "n": len(items),
                "budget_observed_n": len(observed),
                "over_budget_count": over,
                "over_budget_rate": rate(over, len(observed)),
                **{f"median_{field}": median([row.get(field) for row in observed]) for field in BUDGET_FIELDS},
            }
    return out


def context_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, items in sorted(group_rows(rows).items()):
        efficiencies = []
        for row in items:
            ctx = as_float(row.get("retrieved_context_length"))
            speedup = as_float(row.get("official_oracle_best_speedup_vs_eager_p50"))
            if math.isfinite(ctx) and ctx > 0 and math.isfinite(speedup):
                efficiencies.append(speedup / (ctx / 1000.0))
        out[group] = {
            "n": len(items),
            "median_retrieved_context_length": median([row.get("retrieved_context_length") for row in items]),
            "median_oracle_speedup_per_1k_context_length": median(efficiencies),
        }
    return out


def anytime_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = set()
    for row in rows:
        for key in row:
            if key.startswith("official_anytime_") and key.endswith("s_latency_p50_ms"):
                fields.add(key)
    return sorted(fields, key=lambda x: as_float(x.split("official_anytime_", 1)[1].split("s_", 1)[0]))


def anytime_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fields = anytime_fields(rows)
    for field in fields:
        cutoff = field.split("official_anytime_", 1)[1].split("s_", 1)[0]
        speedup_field = f"official_anytime_{cutoff}s_speedup_vs_eager_p50"
        out[cutoff] = {}
        for group, items in sorted(group_rows(rows).items()):
            latencies = [row.get(field) for row in items if finite(row.get(field))]
            speedups = [row.get(speedup_field) for row in items if finite(row.get(speedup_field))]
            out[cutoff][group] = {
                "hidden_correct_best_count": len(latencies),
                "median_latency_p50_ms": median(latencies),
                "median_speedup_vs_eager_p50": median(speedups),
            }
    return out


def time_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for group, items in sorted(group_rows(rows).items()):
        out[group] = {
            "median_time_to_correct_s": median([row.get("official_time_to_correct_s") for row in items]),
            "median_time_to_best_s": median([row.get("official_time_to_best_s") for row in items]),
        }
    return out


def task_penalties(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    correct_by_task: dict[str, list[float]] = defaultdict(list)
    all_correct: list[float] = []
    for row in rows:
        lat = oracle_latency(row)
        if math.isfinite(lat) and lat > 0:
            correct_by_task[str(row.get("task_id", ""))].append(lat)
            all_correct.append(lat)
    global_penalty = max(all_correct) * 1.5 if all_correct else math.nan
    penalties = {
        task_id: max(vals) * 1.5
        for task_id, vals in correct_by_task.items()
        if vals
    }
    return penalties, global_penalty


def sensitivity_latency_getter(penalties: dict[str, float], global_penalty: float):
    def get(row: dict[str, Any]) -> float:
        lat = oracle_latency(row)
        if math.isfinite(lat) and lat > 0:
            return lat
        return penalties.get(str(row.get("task_id", "")), global_penalty)

    return get


def sensitivity_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    penalties, global_penalty = task_penalties(rows)
    getter = sensitivity_latency_getter(penalties, global_penalty)
    out = {
        "penalty_rule": "Use official oracle-best p50 when present; otherwise use 1.5x the worst hidden-correct oracle p50 for the same task, falling back to 1.5x the global worst hidden-correct oracle p50.",
        "global_penalty_latency_p50_ms": global_penalty,
        "comparisons": {},
    }
    for baseline in BASELINES:
        deltas = paired_deltas(rows, baseline, latency_getter=getter)
        out["comparisons"][baseline] = summarize_deltas(deltas)
    return out


def build_report(rows: list[dict[str, Any]], bootstrap_iters: int, bootstrap_run: int) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "input_run_count": len(rows),
        "primary_metric": "official_oracle_best_latency_p50_ms",
        "primary_paired_log_latency": primary_comparisons(rows, bootstrap_iters, bootstrap_run),
        "correctness": correctness_stats(rows),
        "time_to_correct_best": time_summary(rows),
        "budget": budget_summary(rows),
        "context_efficiency": context_summary(rows),
        "anytime": anytime_summary(rows),
        "sensitivity": sensitivity_analysis(rows),
        "notes": [
            "Legality is not judged here; pilot automatic statistics assume candidates are legal until manual review.",
            "Budget and context are post-hoc observations and do not affect candidate generation.",
            "Official/oracle/anytime fields require analyze_results.py input that has merged evaluate_official.py output.",
        ],
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return "" if value is None else str(value)


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# H20 Formal Statistics",
        "",
        f"- input_run_count: {report['input_run_count']}",
        f"- primary_metric: `{report['primary_metric']}`",
        "",
        "## Primary Paired Log Latency",
        "",
    ]
    primary_rows = []
    for baseline, data in report["primary_paired_log_latency"].items():
        ci = data.get("bootstrap_task_95ci", {})
        primary_rows.append(
            {
                "baseline": baseline,
                "paired_block_count": data.get("paired_block_count"),
                "mean_delta_log_latency": data.get("mean_delta_log_latency"),
                "mean_latency_reduction_pct": data.get("mean_latency_reduction_pct"),
                "ci_low_change_pct": ci.get("ci_low_latency_change_pct"),
                "ci_high_change_pct": ci.get("ci_high_latency_change_pct"),
            }
        )
    lines.append(table(primary_rows, ["baseline", "paired_block_count", "mean_delta_log_latency", "mean_latency_reduction_pct", "ci_low_change_pct", "ci_high_change_pct"]))

    lines.extend(["", "## Correctness", ""])
    correctness_rows = []
    for group, data in report["correctness"]["group_pass_rates"].items():
        correctness_rows.append({"group": group, **data})
    lines.append(table(correctness_rows, ["group", "n", "pass_count", "pass_rate", "wilson_95ci_low", "wilson_95ci_high"]))

    lines.extend(["", "## McNemar A3 vs Baselines", ""])
    mcnemar_rows = []
    for baseline, data in report["correctness"]["paired_mcnemar"].items():
        mcnemar_rows.append({"baseline": baseline, **data})
    lines.append(table(mcnemar_rows, ["baseline", "paired_block_count", "a3_pass_baseline_fail", "baseline_pass_a3_fail", "mcnemar_exact_pvalue"]))

    lines.extend(["", "## Time To Correct/Best", ""])
    time_rows = [{"group": group, **data} for group, data in report["time_to_correct_best"].items()]
    lines.append(table(time_rows, ["group", "median_time_to_correct_s", "median_time_to_best_s"]))

    lines.extend(["", "## Budget By Group", ""])
    budget_rows = [{"group": group, **data} for group, data in report["budget"]["group"].items()]
    lines.append(table(budget_rows, ["group", "n", "budget_observed_n", "over_budget_count", "over_budget_rate", "median_budget_candidate_count", "median_budget_compile_attempts", "median_budget_correctness_runs", "median_budget_benchmark_runs", "median_budget_harness_runs"]))

    lines.extend(["", "## Context Efficiency", ""])
    context_rows = [{"group": group, **data} for group, data in report["context_efficiency"].items()]
    lines.append(table(context_rows, ["group", "n", "median_retrieved_context_length", "median_oracle_speedup_per_1k_context_length"]))

    lines.extend(["", "## Anytime", ""])
    anytime_rows = []
    for cutoff, groups in report["anytime"].items():
        for group, data in groups.items():
            anytime_rows.append({"cutoff_s": cutoff, "group": group, **data})
    lines.append(table(anytime_rows, ["cutoff_s", "group", "hidden_correct_best_count", "median_latency_p50_ms", "median_speedup_vs_eager_p50"]) if anytime_rows else "No anytime fields found.")

    lines.extend(["", "## Sensitivity", "", report["sensitivity"]["penalty_rule"], ""])
    sens_rows = []
    for baseline, data in report["sensitivity"]["comparisons"].items():
        sens_rows.append({"baseline": baseline, **data})
    lines.append(table(sens_rows, ["baseline", "paired_block_count", "mean_delta_log_latency", "mean_latency_reduction_pct"]))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True, help="JSON produced by analyze_results.py.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-run", type=int, default=0)
    args = parser.parse_args()

    analysis = read_json(Path(args.analysis_json))
    rows = analysis.get("runs", [])
    if not isinstance(rows, list):
        raise TypeError("analysis JSON must contain a list at key 'runs'")

    report = build_report(rows, args.bootstrap_iters, args.bootstrap_run)
    write_json(Path(args.out_json), report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown(report), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
