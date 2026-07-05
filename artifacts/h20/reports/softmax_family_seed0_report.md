# H20 MVP Experiment Report

- run_records: 20
- csv: `artifacts/h20/reports/softmax_family_seed0_report.csv`
- json: `artifacts/h20/reports/softmax_family_seed0_report.json`

## Group summary

| group | op_family | runs | compile_success_rate | hidden_correctness_pass_rate | correct_and_faster_rate_vs_torch_compile | median_speedup_vs_torch_compile_p50 | median_iterations_to_first_correct | median_wall_time_s | median_invalid_compile_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_no_kb | softmax | 5 | 1 | 1 | 1 | 2.194 | 1 | 3.183 | 0 |
| A1_plain_rag | softmax | 5 | 1 | 1 | 1 | 2.193 | 1 | 2.467 | 0 |
| A2_rulebook | softmax | 5 | 1 | 1 | 1 | 2.38 | 1 | 2.408 | 0 |
| A3_ecc_kb | softmax | 5 | 1 | 1 | 1 | 2.16 | 1 | 2.415 | 0 |

## Interpretation checklist

- Compare A3 against A0/A1/A2 under the same task, seed, and budget.
- Treat hidden correctness as a hard gate before reading performance metrics.
- Check whether A3 reduces iterations, wall time, invalid compile attempts, and context cost.
- For self-evolution, compare A3(v1) against A3(v0) on Round-2 held-out tasks.
