# H20 MVP 验证实验协议

## 1. 实验目标

H20 只是第一阶段验证载体。MVP 要验证的是 ECC-KB 方法本身是否让任意 Coding Agent 在相同预算下：

- 更快得到编译通过的 kernel；
- 更高概率通过强正确性测试；
- 更高概率快于 PyTorch eager 和 `torch.compile` / Inductor baseline；
- 使用更少 token、wall-clock time、GPU benchmark 次数和迭代次数；
- 能把第一轮经验沉淀成第二轮有效知识。

## 2. 实现框架选择

第一阶段建议使用 **Triton + PyTorch reference + Python harness**。

选择 Triton 的理由：

- 代码量较小，适合 Agent 生成和修改；
- 支持 Nvidia，也可迁移到 AMD ROCm Triton 和 Triton-Ascend 等路径；
- 参数空间明确，便于把 knobs 结构化；
- 与 PyTorch reference、`torch.compile` baseline 和 torch CUDA event 计时集成成本低；
- 不把方法绑定到 CUDA C++ 细节。

若后续目标是 CUDA C++、HIP、AscendC、TileLang 或 MLIR，ECC-KB 的 schema、ContextPacket、门控和演化机制保持不变，只替换 backend adapter。

## 3. 固定环境记录

每个 run 必须记录：

```text
hardware:
  device_name: runtime detected
  compute_capability: runtime detected
  total_memory: runtime detected
software:
  python, pytorch, triton, cuda, driver
runtime:
  CUDA_VISIBLE_DEVICES
  clock/persistence setting if available
  warmup/repeat counts
  random seeds
repo:
  git commit
  knowledge base version
  harness version
```

不要在方法中硬编码 H20 的具体硬件参数。用 runtime fingerprint 记录即可。

## 4. 算子任务集

MVP 选择四类中等复杂度算子：row reduction、softmax、layernorm、matmul variants。它们共同覆盖 reduction、数值稳定、mask/boundary、tile tuning、memory bandwidth、compute intensity 和 profile 归因。

任务规模见 `experiments/h20/operators.md`。

## 5. 实验组

固定同一 Coding Agent、同一任务顺序、同一最大迭代数、同一 token 上限和同一 harness，比较四组：

1. `A0_no_kb`：无知识库，只给 OpSpec 和 harness 指令。
2. `A1_plain_rag`：普通文档 chunk/向量召回，返回段落文本。
3. `A2_rulebook`：简单规则库/经验库，返回静态规则和 prompt tips。
4. `A3_ecc_kb`：ECC-KB 返回 ContextPacket，使用 quarantine/promotion 写回。

每个任务至少跑 3 个随机种子或 3 次独立 agent session。若预算有限，先对 12 个任务跑 2 次，筛出最能区分方法的任务后扩展到 24–32 个任务。

## 6. Agent 接入方式

Agent 通过文件协议接入，不依赖内部工具：

```text
workspace/
  task.json              # OpSpec + goal + budget
  context_packet.json    # 由 KB 或 baseline 生成
  candidate.py           # Agent 写入 Triton kernel
  run.sh                 # harness 固定命令
  results.json           # harness 输出
  trace.jsonl            # 每次迭代事件
```

Agent 每轮只允许修改候选代码和说明，不允许修改 reference、hidden tests、benchmark harness、metrics 脚本。

## 7. 正确性门控

正确性分四层：

1. smoke：1–2 个小 shape，快速发现编译/接口错误；
2. quick：公开 shape sweep，覆盖 dtype、边界和非 2 次幂；
3. hidden：Agent 不可见的 shape/dtype/stride seeds；
4. robustness：数值稳定、determinism、non-contiguous、broadcast/stride/alias 按算子适用性开启。

性能测试必须在 hidden correctness 通过后执行。

## 8. 性能测量

默认使用 torch CUDA event 计时：

```text
warmup: 100
repeats: 500
report: p50, p95, mean, std, min
compare: PyTorch eager, torch.compile if applicable, known Triton baseline if available
```

可选使用 Nsight Compute / Triton profiler 提取简化 profile symptom。不要把全量 profiler 输出直接给 Agent。由 harness 归一化为：

```text
profile_summary:
  dominant_symptom: memory_bound | compute_bound | occupancy_limited | launch_bound | unknown
  evidence: selected counters or derived ratios
  candidate_actions: linked capsule ids
```

## 9. 成功标准

MVP 不要求每个任务都达到专家 kernel 性能。建议采用以下最小成功标准：

```text
A3 vs A0:
  +10% absolute hidden correctness pass rate, or
  -25% median iterations to first correct kernel
A3 vs A1/A2:
  higher or equal hidden correctness with lower token/time, and
  higher median speedup among correct kernels
Self-evolution:
  Round-2 held-out tasks show statistically visible improvement over Round-1 frozen KB
```

更强目标：A3 在 24 个任务上相对 A0/A1/A2 同时降低 token 20%、wall time 20%、iterations 25%，且 correct-and-faster rate 最高。

## 10. 实验流程

```text
prepare initial KB
  -> freeze KB version v0
  -> run A0/A1/A2/A3 on Round-1 tasks
  -> collect traces and benchmark/profile evidence
  -> ECC-KB quarantine ingest
  -> promotion gate
  -> freeze KB version v1
  -> run A3(v1) and baselines on Round-2 held-out variants
  -> compare metrics and knowledge utility
```

## 11. 判断知识进化有效

知识库进化有效必须同时满足：

- Round-2 中被召回的 promoted capsule 有较高实际采用率；
- 使用这些 capsule 的任务减少无效编译/错误修复/低收益 tuning；
- 失败 capsule 阻止了重复错误；
- v1 相比 v0 在 held-out shape/dtype/变体上改善，而不是只记住 Round-1 任务。

如果 v1 只增加上下文长度而不降低迭代或 token，说明演化机制没有形成可复用知识。
