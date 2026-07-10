# H20 Codex Hook 记录协议

本文档说明正式 H20 实验中如何使用 Codex Hook 记录子代理运行指标。当前实验设定是：主代理只负责派遣和汇总，不进入 tuning budget，也不作为受限优化主体。因此 active Hook 只绑定 `SubagentStart`、`SubagentStop`、`PreToolUse` 和 `PostToolUse`，不默认启用 `SessionStart` / `Stop`。

Hook 的定位是旁路观测器：记录时间、candidate 快照、candidate 修改尝试、compile/correctness/benchmark 调用、以及 harness 结果中的 p50/speedup 辅助字段。它不改变四个实验组的输入、检索内容、harness 或 evaluator，因此不应成为某一组额外能力来源。

## 1. 记录目标

正式实验需要同时回答性能、正确性和成本问题。因此 Hook 至少记录以下信息：

| 指标 | 记录方式 |
| --- | --- |
| subagent wall time | `SubagentStart` / `SubagentStop` 记录每个子代理时间 |
| candidate 快照 | `PostToolUse` 在 candidate 修改后快照；timer 在 120/240/480s 快照；按 sha256 去重 |
| candidate 修改次数 | `PreToolUse` 判断命令是否尝试修改 `candidate.py`，计入 `candidate_edit_attempts` |
| compile attempts | `PreToolUse` 对命令做 stage/关键字匹配 |
| correctness runs | `PreToolUse` 对命令做 stage/关键字匹配 |
| benchmark runs | `PreToolUse` 对命令做 stage/关键字匹配 |
| final/oracle p50 | opportunistic 解析 `results.json` / `benchmark.json`；正式结果仍由 evaluator 重测 |

主代理没有 budget 限制时，不应把 `SessionStart -> Stop` 的端到端时间作为主 cost 指标。否则会把主代理调度、自然语言汇总、人工交互等待等非受控开销混入组间比较。

## 2. 为什么需要 PostToolUse 快照

`PreToolUse` 发生在工具执行之前，因此它只能记录“即将修改 candidate”的意图和旧 candidate 状态。真正反映工具副作用的 candidate 版本应在 `PostToolUse`、`SubagentStop` 或定时器中保存。

因此本仓库实现的 active 策略是：

```text
SubagentStart:
  - 记录子代理开始时间
  - 保存初始快照
  - 启动 120/240/480s 定时快照进程

PreToolUse:
  - 统计 candidate 修改尝试、compile/correctness/benchmark/harness 调用尝试
  - 如果将要修改 candidate.py，保存 before_candidate_edit 快照

PostToolUse:
  - 如果工具可能修改了 candidate.py，保存 after_candidate_edit 快照
  - 如果工具可能运行了 harness，解析 results.json / benchmark.json

SubagentStop:
  - 保存最终快照
  - 解析最终 results.json / benchmark.json
  - 写 summary.json
```

`h20_codex_hook.py` 内部仍保留 `SessionStart` / `Stop` 分支作为兼容逻辑；只要 `hooks.json` 不声明这些事件，它们就不会在正式实验中触发。

## 3. any-time 指标的正确计算方式

Hook 的 120/240/480s timer 只保存 candidate 快照，不主动跑 benchmark。原因是：如果 Hook 在固定时间点额外运行 benchmark，会改变 Agent 运行期间的 GPU 使用、wall time 和 benchmark budget，使实验组比较不公平。

正式 any-time p50/speedup 应使用离线 evaluator：

```text
1. 收集 .codex_h20_metrics/candidates/*.py
2. 根据 trace.jsonl 中的 timestamp 选择 120/240/480s 时刻已经出现的 candidate
3. 用冻结 evaluator 对这些 candidate 统一跑 hidden correctness 和 benchmark
4. 得到 any-time best hidden-correct p50 / speedup；cheating/illegal 由主代理事后人工检查
```

## 4. 对 harness 命令的要求

Hook 可以用正则推断命令语义，但正式实验应降低歧义。推荐所有 harness 命令显式带 stage：

```bash
./run.sh --stage compile
./run.sh --stage smoke
./run.sh --stage quick
./run.sh --stage hidden
./run.sh --stage benchmark
```

默认 Hook 正则会识别这些 stage。若实际 runner 使用其他命令名，应通过环境变量覆盖：

```bash
H20_COMPILE_RE
H20_CORRECTNESS_RE
H20_BENCHMARK_RE
H20_HARNESS_RE
```

## 5. 输出文件语义

默认输出目录为 `.codex_h20_metrics/`。

| 文件 | 语义 |
| --- | --- |
| `trace.jsonl` | append-only 事件流，用于审计和后处理 |
| `state.json` | Hook 累积状态，便于中途恢复 |
| `summary.json` | run-level 汇总，方便批量读取 |
| `hook_errors.jsonl` | Hook 自身异常；Hook 异常不应中断实验 |
| `candidates/*.py` | 按 sha256 去重的 candidate 快照 |

`summary.json` 中建议正式读取 `agents[*].wall_time_s` 作为子代理 wall time；`session_wall_time_s` 在默认 subagent-only Hook 配置下为空或不作为主指标。

`summary.json` 中可能保留 `agent_final_legal_p50_ms` / `oracle_best_legal_p50_ms` 等历史命名的辅助字段；pilot 自动统计不把它们当作 legality 指标。最终论文/报告中的 official correctness/performance 指标必须由独立 evaluator 重跑，cheating/illegal 由主代理事后人工检查。

## 6. 最小校准检查

正式跑 720 个 workspace 前，先跑 1 个 calibration workspace，检查：

1. `trace.jsonl` 中是否出现 `SubagentStart`、`PreToolUse`、`PostToolUse`、`SubagentStop`；
2. `trace.jsonl` 中不应出现 active `SessionStart` / `Stop` 记录，除非你手动启用了 root-session hook；
3. `candidate_edit_attempts` 是否接近人工观察到的 candidate 修改次数；
4. compile/correctness/benchmark runs 是否与 harness 日志一致；
5. `candidates/` 是否保存了去重后的 candidate；
6. timer 是否产生 `subagent_t120s` 等快照；
7. `summary.json` 是否能被批量评估脚本读取。

如果校准失败，先修正 matcher 或 stage 正则，再开始正式实验。
