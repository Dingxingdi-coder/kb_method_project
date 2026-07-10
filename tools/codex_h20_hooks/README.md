# Codex H20 Hook Metrics Collector

本目录提供 H20 正式实验使用的 Codex Hook 观测模块。当前默认配置只记录子代理生命周期：主代理只负责派遣和汇总，不进入 tuning budget，也不作为受限优化主体。

Hook 的定位是旁路记录器：记录时间、candidate 快照、candidate 修改尝试、compile/correctness/benchmark 调用、以及 harness 结果中的 p50/speedup 字段；它不修改 `candidate.py`，不修改 harness，也不向 Agent 注入额外上下文。

## 文件

```text
tools/codex_h20_hooks/
  hooks.json             # Codex hooks.json 模板，只启用 subagent 相关事件
  h20_codex_hook.py      # 单文件 Hook 处理器
```

## 安装到单个 workspace

在仓库 checkout 或 workspace 中执行：

```bash
mkdir -p .codex
cp tools/codex_h20_hooks/hooks.json .codex/hooks.json
```

然后启动 Codex，并在 Codex 里使用 `/hooks` review/trust 该 Hook 配置。仓库的 `.gitignore` 忽略 `.codex/`，因此每个实验 workspace 可以拥有自己的本地 Hook 配置，不会污染版本库。

如果 Codex 不是从 Git 仓库内部启动，`hooks.json` 里的 `git rev-parse --show-toplevel` 会失败。此时应把 `hooks.json` 中的命令改成 `h20_codex_hook.py` 的绝对路径，或者在实验 runner 里生成 workspace-local `.codex/hooks.json`。

## Active Hook 范围

默认 `hooks.json` 只启用：

```text
SubagentStart
PreToolUse
PostToolUse
SubagentStop
```

不启用：

```text
SessionStart
Stop
```

原因是主代理没有正式 budget 限制。如果把 `SessionStart -> Stop` 纳入主指标，会把主代理调度、自然语言汇总、人工交互等待等非受控开销混入实验结果。`h20_codex_hook.py` 内部仍保留 root-session 分支，供将来需要端到端观测时手动启用。

## 推荐环境变量

```bash
export H20_RUN_ID="task001_A3_seed0"
export H20_GROUP="A3_ecc_kb"
export H20_TASK="row_softmax_4096x1024_bf16"
export H20_WORKSPACE_DIR="$PWD"
export H20_CANDIDATE_PATH="$PWD/candidate.py"
export H20_METRICS_DIR="$PWD/.codex_h20_metrics"
export H20_HOOK_SNAPSHOT_SECONDS="120,240,480"
```

可选正则覆盖：

```bash
export H20_COMPILE_RE='(--stage[= ]compile|compile|triton|jit|nvcc)'
export H20_CORRECTNESS_RE='(--stage[= ](smoke|quick|hidden|correctness)|pytest|correctness)'
export H20_BENCHMARK_RE='(--stage[= ](benchmark|bench|perf)|benchmark|latency|speedup)'
export H20_HARNESS_RE='(^| )(./)?run.sh( |$)|harness'
```

正式实验中建议让 harness 命令显式带 stage，例如：

```bash
./run.sh --stage compile
./run.sh --stage quick
./run.sh --stage hidden
./run.sh --stage benchmark
```

这样 Hook 统计 compile/correctness/benchmark attempts 时不需要猜测命令语义。

## 输出

默认输出到 workspace 下的 `.codex_h20_metrics/`：

```text
.codex_h20_metrics/
  trace.jsonl           # Hook 事件流，可审计
  state.json            # 累积状态
  summary.json          # run-level 汇总
  hook_errors.jsonl     # Hook 自身异常，不中断实验
  candidates/
    0001_*.py           # 去重后的 candidate 快照
```

`summary.json` 中的核心字段包括：

```text
agents[*].wall_time_s
totals.candidate_edit_attempts
totals.compile_attempts
totals.correctness_runs
totals.benchmark_runs
candidate_count
agent_final_legal_p50_ms  # legacy auxiliary field name
oracle_best_legal_p50_ms  # legacy auxiliary field name
observed_best_unfiltered_p50_ms
```

默认 subagent-only 配置下，正式 wall time 主指标应使用 `agents[*].wall_time_s`。`session_wall_time_s` 可能为空，不进入主表。

`agent_final_legal_p50_ms` 和 `oracle_best_legal_p50_ms` 是历史命名的辅助字段，不作为 pilot legality 指标。pilot 自动统计阶段先假设 candidate legal；cheating/illegal 由主代理事后人工检查。上述辅助字段只会在 harness 的 `results.json` 或 `benchmark.json` 中能解析出以下信息时产生：

1. latency p50；
2. correctness pass 或 hidden correctness pass；
3. legality pass / legal / is_legal；
4. result file 不比当前 `candidate.py` 更旧。

如果 legality 字段缺失，Hook 不会把该 observation 算作 legal performance。这样可以避免把未判定合法性的 candidate 错记为 official best。

## 重要限制

1. `PreToolUse` 只能可靠记录“将要修改 candidate.py”的尝试；真正的文件快照主要由 `PostToolUse` 和定时器记录。
2. Hook 不会在 120/240/480s 主动跑 correctness 或 benchmark。它只保存 candidate 快照。正式 any-time p50/speedup 应由冻结 evaluator 离线重放这些快照得到。
3. Codex 的 shell 工具名和命令格式可能随版本变化。正式实验前必须用一个 calibration workspace 检查 `trace.jsonl` 中的 `tool_name` 和计数是否符合预期。
4. Hook 记录的是观测数据，不是最终裁判。official final/oracle 指标仍应由独立 evaluator 重跑 hidden correctness、legality filter 和 benchmark。
