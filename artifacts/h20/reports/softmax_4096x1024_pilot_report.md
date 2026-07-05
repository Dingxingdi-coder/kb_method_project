# H20 MVP Experiment Report

- run_records: 8
- csv: `artifacts/h20/reports/softmax_4096x1024_pilot_report.csv`
- json: `artifacts/h20/reports/softmax_4096x1024_pilot_report.json`

## Group summary

| group | op_family | runs | compile_success_rate | hidden_correctness_pass_rate | correct_and_faster_rate_vs_torch_compile | median_speedup_vs_torch_compile_p50 | median_iterations_to_first_correct | median_wall_time_s | median_invalid_compile_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_no_kb | reduction | 1 | 1 | 0 | 0 |  |  | 0.2513 | 0 |
| A0_no_kb | softmax | 1 | 1 | 1 | 0 | 0 | 1 | 1.237 | 0 |
| A1_plain_rag | reduction | 1 | 1 | 0 | 0 |  |  | 0.2008 | 0 |
| A1_plain_rag | softmax | 1 | 1 | 1 | 0 | 0 | 1 | 0.3217 | 0 |
| A2_rulebook | reduction | 1 | 1 | 0 | 0 |  |  | 0.2522 | 0 |
| A2_rulebook | softmax | 1 | 1 | 1 | 0 | 0 | 1 | 1.04 | 0 |
| A3_ecc_kb | reduction | 1 | 1 | 0 | 0 |  |  | 0.187 | 0 |
| A3_ecc_kb | softmax | 1 | 1 | 1 | 0 | 0 | 1 | 0.4074 | 0 |

## Interpretation checklist

- Compare A3 against A0/A1/A2 under the same task, seed, and budget.
- Treat hidden correctness as a hard gate before reading performance metrics.
- Check whether A3 reduces iterations, wall time, invalid compile attempts, and context cost.
- For self-evolution, compare A3(v1) against A3(v0) on Round-2 held-out tasks.
