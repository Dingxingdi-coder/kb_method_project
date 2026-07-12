# H20 正式实验方案

核心原则：不要只验证 "A3 平均更快"，而要验证三个更强的问题：

- A3 是否在等预算下稳定优于普通 RAG；
- A3 的收益是否来自 ECC-KB 的结构化机制，而不是偶然样本或上下文长度；
- KB 自进化 v1 是否能在严格 held-out 任务上继续提升。

## 0. 预注册 Claims

正式实验应提前写清楚最终要证明的结论。建议拆成四个 claim。

### 0.1 有效性 Claim

在相同 Agent workflow、相同 H20 harness、相同事后预算审计阈值下，`A3_ecc_kb` 相比 `A0`/`A1`/`A2` 有更好的 hidden-correct 性能。

主指标建议使用：

- oracle best p50 latency；
- 或 speedup。

### 0.2 上下文效率 Claim

A3 不是靠塞更多 context 获得效果，而是在 retrieved context length 更少或相同上限下取得更好结果。

### 0.3 成本 Claim

报告：

- time-to-correct；
- time-to-best；
- any-time 曲线。
- budget usage：candidate 数、compile/correctness/benchmark/harness 调用数、wall time；
- over-budget rate：按预注册审计阈值标记后的超预算比例。

定义：

- 正确 candidate：从 Agent 开始处理任务，到第一次生成一个 correctness pass 的实现所用时间。
- 最佳 candidate：在一次完整 Agent 运行中产生的所有 hidden-correct candidates 中，最终 evaluator 重新 benchmark 后 p50 最低的 candidate。

### 0.4 自进化 Claim

KB v1 必须只从 evolution set 的经验中 promotion 得到，然后在 held-out set 上比较 `A3(v1)` 和 `A3(v0)`。

不能用同一批任务既产经验又评估 v1。

## 1. 任务数、Run 数和算子类别

主实验建议使用：

```text
36 tasks x 5 runs x 4 groups = 720 workspaces
```

### 1.1 推荐任务分布

正式任务集不要只放容易出效果的算子。建议覆盖 6 类，每类 6 个任务，共 36 个任务。

| 类别 | 目的 | 示例任务 |
| --- | --- | --- |
| Pointwise / fused memory-bound | 检验基础 fusion、访存优化、向量化 | bias+activation、GELU/SILU、residual add、broadcast elementwise、gated multiply、clamp/mul/add |
| Reduction | 检验 warp/block reduction、跨 block 归约、边界处理 | sum、max、mean、L2 norm、argmax、row/column reduction |
| Softmax / logsumexp / attention primitive | 检验数值稳定性和分阶段优化 | row softmax、masked softmax、causal softmax、logsumexp、softmax backward、cross entropy forward |
| Normalization | 检验多阶段统计量、精度和融合 | LayerNorm、RMSNorm、GroupNorm、BatchNorm inference、mean/variance fusion、norm backward 简化版本 |
| Matmul-like / tensor core / epilogue | 检验 compute-bound 和 tensor core 使用 | GEMM+bias、batched matmul、小 K matmul、matvec、linear+activation、skinny GEMM |
| Layout / indexing / irregular memory | 检验非连续访存和 irregular pattern | transpose、permute/copy、gather、scatter-add、embedding lookup、slice/concat 类任务 |

### 1.2 Shape Case 设计

每个任务内部至少包含：

1. public correctness shapes：Agent 可见，用于开发调试；
2. public benchmark shapes：Agent 可见，用于 tuning；
3. hidden correctness shapes：Agent 不可见，用于正式 correctness；
4. hidden benchmark shapes：Agent 不可见，或至少不可直接调参，用于最终评估。

shape 设计应覆盖：

- small / medium / large；
- power-of-two / non-power-of-two；
- 整除 / 非整除；
- fp16 / bf16 / fp32；
- 必要时的 contiguous / non-contiguous。

softmax、normalization、reduction 类任务应包含数值极端 case，例如大正数、大负数、mask 全空或近似全空、长度为 1、长度不是 32/64/128 倍数等。

## 2. 实验组

建议保留原四组：

```text
A0_prompt
A1_raw_corpus_vector_rag
A2_kb_vector_rag
A3_ecc_kb
```

## 3. Budget 审计和环境控制

### 3.1 Wall Time

expanded pilot 不在子代理运行过程中强制 wall-time hard cap，不打断子代理；wall time 只从 Hook / agent session 日志中事后读取，并按预注册阈值标记是否超预算。

建议预注册统一 wall-time 审计阈值：

```text
每个 workspace wall time 审计阈值 = 480s
```

此外必须从日志中重构 any-time 指标：

| 截止点 | 指标 |
| --- | --- |
| 120s | 当前 best hidden-correct candidate 的 p50 / speedup |
| 240s | 当前 best hidden-correct candidate 的 p50 / speedup |
| 480s | 当前 best hidden-correct candidate 的 p50 / speedup |

这样可以回答：A3 是更慢但最终更好，还是在相同截止点下也更好。超出审计阈值的 workspace 在结果表中标记，不在运行时中断。

### 3.2 Tuning Budget

expanded pilot 不通过 prompt 限制子代理 tuning 行为，也不在运行时限制 harness 调用次数。正式分析只做事后审计：从 Hook / trace / harness 输出读取 candidate 数、compile/correctness/benchmark/harness 调用数，并按预注册阈值标记超预算。

这不是“不记录 budget”。budget 相关内容仍然是 cost 指标：实验结束后统一统计、标记、汇总，并可用于组间比较、sensitivity analysis 或剔除规则；它们只是不作为子代理运行时的 prompt 约束、停止条件或干预信号。

| 审计项 | 建议阈值 |
| --- | --- |
| max modification numbers | 12-16 |
| max compile attempts | 16-24 |
| max correctness runs | 16-24 |
| max benchmark runs | 8-12 |
| max wall time | 600s，所有组相同 |

原因：不同组可能通过不同数量的 benchmark calls 获得不同程度的 tuning advantage。best p50 尤其容易受 benchmark 调用次数影响。因此这些量必须报告并标记超预算，但 pilot 阶段不在子代理运行中干预。

Autotune 的正式比较口径仍然是 hidden correctness 通过后的 benchmark p50/p95。硬件 profiling 只作为可选的 post-correctness 诊断证据，用来解释 bottleneck 或指导下一轮修改；不要求每个 candidate 都跑 profiling，也不把 profiler counter 作为主统计指标。

事后审计命令示例：

```bash
python experiments/h20/audit_budgets.py \
  --runs artifacts/h20/expanded_pilot_runs \
  --out artifacts/h20/expanded_pilot_budget_audit.json \
  --max-agent-wall-time-s 480 \
  --max-candidates 16 \
  --max-compile-attempts 24 \
  --max-correctness-runs 24 \
  --max-benchmark-runs 12
```

正式性能建议使用两个版本：

1. `final_p50`：Agent 最终提交的 hidden-correct candidate 的 p50；
2. `oracle_best_p50`：一次完整 Agent 运行中生成过的所有 hidden-correct candidates 中 p50 最低者。

前者衡量 Agent 最终选择能力，后者衡量生成/搜索能力。主指标可以用 `oracle_best_p50`，但必须同时报告 final，避免过度美化。

### 3.3 Context Budget

所有检索组使用同一最大上限。

| 组别 | retrieved context length cap |
| --- | --- |
| A1 | 12k |
| A2 | 12k |
| A3 | 12k 总上限，分 stage cap |

正式报告中要区分：

1. retrieved context length；
2. agent wall time。

pilot 自动统计 retrieved context length。

### 3.4 GPU 分配和 Benchmark 控制

H20 benchmark 容易受 GPU 共享、温度、clock、进程干扰影响。建议：

1. 每个 workspace 独占一张 H20，不允许多个 Agent 同时在同一张 GPU 上跑 benchmark。
2. 禁用 MPS/MIG 共享，或者至少保证 benchmark 阶段独占。
3. 记录 GPU ID、driver、CUDA、PyTorch/Triton 版本、power cap、clock policy、温度区间。
4. 同一个 task-run 的不同实验组应尽量在同一类 GPU、同一环境、相近时间窗口内完成。
5. 实验顺序要 randomize/block，不能总是 A0 先跑、A3 后跑。
6. 最终 benchmark 最好由独立 evaluator 统一重跑，而不是直接使用 Agent 自己日志里的最佳数字。
7. 对每个 final/best candidate 做 warmup 后重复 benchmark，报告 p50，同时保存 p95 / tail latency。

正式 benchmark 推荐流程：

```text
Agent 运行期间可以调用 correctness 和 benchmark 进行 tuning；这些调用只进入事后预算审计。
实验结束后，evaluator 收集所有 candidate code。
evaluator 在干净进程里重新 benchmark final candidate 和 oracle-best candidate。
正式表格使用 evaluator 重测结果，而不是 Agent 内部观察值。
```

## 4. Legality / Cheating 判定

这个部分必须在正式实验前写成 rubric，并作为任务 spec 的一部分冻结。否则高层 API fallback 会严重影响结论。

### 4.1 Kernel-Authoring Mode

主实验建议定义为 kernel-authoring mode：

```text
候选实现必须通过自写 CUDA/Triton/custom kernel 或等价低层 kernel 实现目标算子。
不允许在 submitted function 的有效路径上调用 PyTorch/ATen/cuBLAS/cuDNN 等高层实现来完成目标语义或关键子语义。
```

也就是说，主实验评价的是 Agent 写特化算子的能力，不是调用已有高层库拼装的能力。

可以另开 secondary library-composition mode，允许调用 vendor primitives，但这应单独报告，不能和 kernel-authoring mode 混在一起。

### 4.2 API 分层

Allowed infrastructure APIs：允许。

- 读取 shape、stride、dtype、device；
- 分配输出 tensor，例如 `torch.empty`、`torch.empty_like`、`new_empty`；
- 编译和加载自定义 extension；
- 调用自写 CUDA/Triton kernel。

Always banned APIs：总是禁止。

- 在 submitted path 中调用 `torch.compile`；
- 直接调用 PyTorch reference；
- 修改 harness；
- 读取 hidden test；
- CPU fallback；
- NumPy/CuPy 高层 fallback；
- 缓存 benchmark 输入输出；
- 用全局状态识别 benchmark。

Target-banned APIs：按任务禁止。

- softmax 任务中，`torch.softmax` 禁止；用 `torch.max` 和 `torch.sum` 组合实现 softmax 也应禁止，因为 max reduction 和 sum reduction 是 softmax 的核心子操作。
- reduction_max 任务中，`torch.max` 禁止。
- LayerNorm 任务中，`torch.mean`、`torch.var`、`torch.nn.functional.layer_norm` 禁止。
- Matmul 任务中，`torch.matmul`、`torch.mm`、`torch.bmm`、`einsum`、cuBLAS 调用禁止。

Review-required APIs：需要人工判定。

- `contiguous()`、`reshape()`、`permute()`、`as_strided()`；
- CUB/Thrust/CUTLASS template；
- Triton library function；
- 某些 ATen metadata op。

判定规则：如果它们触发数据 copy、完成目标主要计算，或者实质上把核心算子交给高层库，就判 illegal；如果只是 metadata/view 或 kernel 内部低层 primitive，可以判 legal。

### 4.3 Softmax Fallback 判定

用 `torch.max` / `torch.sum` 构造 softmax 的 candidate 应判为：

```text
illegal target high-level API fallback
```

即使它 hidden correctness 通过、benchmark 很快，也应在事后人工审查中标注，必要时从 pilot 分析中剔除。

处理方式：

1. pilot 自动统计先假设所有 candidate legal，只计算 hidden correctness、p50/p95、speedup、cost、context、anytime。
2. 实验结束后，主代理人工检查 final candidate 和 oracle-best candidate 是否作弊/illegal。
3. 如果人工检查发现作弊/illegal，将结果作为独立审查备注；必要时剔除对应 candidate 后重算 sensitivity analysis。
4. pilot 阶段不自动统计 cheating/illegal attempt rate。

### 4.4 判定流程

建议使用 deterministic filter + LLM judge + 人工复核。

第一层：静态规则扫描。

扫描 Python、C++、CUDA、Triton 源码，匹配 banned imports、banned calls、ATen native calls、cuBLAS/cuDNN 符号、torch 高层 API、harness 修改、CPU fallback、NumPy/CuPy fallback。

第二层：动态运行时检测。

在 candidate function 执行期间 monkeypatch 或 trace `torch.*`、`torch.nn.functional.*`、ATen op、CUDA library calls。只统计 submitted candidate path，不统计 reference correctness path。对 compiled extension 可以做符号扫描和运行时库调用检测。

第三层：LLM judge + 人工复核。

LLM judge 用于初筛和解释，不作为最终裁判。它输入 task spec、allowed/banned API 列表、candidate code、静态扫描结果、动态 trace，输出 label：legal / illegal / uncertain，并标注可疑代码行。

人工复核建议：

1. 所有 official final candidate 和 oracle-best candidate 都人工复核；
2. 所有静态/动态/LLM flagged candidate 都人工复核；
3. 从未 flagged candidate 中随机抽 10%-20% 复核，用来估计 false negative；
4. reviewer 对 arm blind，不知道代码来自 A0/A1/A2/A3；
5. 至少两名 reviewer 独立标注，冲突由第三人 adjudicate；
6. pilot 阶段只输出独立审查备注，或在必要时作为剔除依据；不把 cheating/illegal rate 作为自动统计指标。

## 5. 统计指标

pilot 自动统计阶段先假设所有 candidate 都是 legal 的，只计算 correctness、performance、cost、context、anytime、category/group comparison 等指标。最后由主代理人工检查 candidate 是否作弊/illegal；人工检查结果作为独立审查备注或必要时的剔除依据，不作为 pilot 自动统计指标。

正式自动指标分为 correctness、performance、cost、context efficiency 四类。

### 5.1 Correctness 指标

| 指标 | 定义 |
| --- | --- |
| public correctness pass rate | Agent 可见 correctness 是否通过 |
| hidden correctness pass rate | hidden correctness 是否通过 |
| first-correct rate | 在预算内是否至少生成过一个 correct candidate |
| final-correct rate | 最终提交 candidate 是否 correct |
| correctness failure reason (case study)| compile fail / runtime fail / numerical mismatch / timeout |

主表使用 hidden correctness pass rate。人工 legality 审查只作为备注或剔除依据。

### 5.2 Performance 指标

| 指标 | 定义 |
| --- | --- |
| final p50 | Agent 最终提交的 hidden-correct candidate 的 p50 |
| oracle best p50 | 固定预算内所有 hidden-correct candidates 中 p50 最低者 |
| speedup vs eager | eager p50 / candidate p50 |
| speedup vs torch.compile | torch.compile p50 / candidate p50 |
| category-level p50 / speedup | 按算子类别聚合 |
| p95 / tail latency | secondary metric |

这里不用指标的原始数值（当然是要记录原始数值），而是使用：

```text
paired log latency improvement of A3 vs A2 on oracle_best_p50
```

对每个 task-run block 计算：

```text
Delta = log(p50_A3) - log(p50_A2)
```

`Delta < 0` 表示 A3 更快。最终报告 `exp(mean Delta) - 1` 或 `exp(median Delta) - 1`，转换成百分比 latency reduction。

A3 不应该只和 A0 比。A0 是无知识 baseline，A2 是最接近的 KB baseline，因此 A3 vs A2 应作为 primary comparison。

### 5.3 Cost 指标

| 指标 | 定义 |
| --- | --- |
| agent wall time | 从 workspace start 到 stop 的总时间 |
| number of candidates | 生成 candidate 数 |
| number of benchmark calls | tuning benchmark 调用数 |
| number of correctness calls | correctness 调用数 |
| budget audit status | 事后审计是否超过预注册阈值 |
| over-budget rate | 按组/类别聚合的超预算 workspace 比例 |

预算相关指标进入 cost 表和组间比较。它们不限制子代理当场继续工作，也不改变 candidate 生成过程。

### 5.4 Context Efficiency 指标

| 指标 | 定义 |
| --- | --- |
| retrieved context length | 检索返回内容长度 |
| speedup per 1k retrieved context length | context efficiency secondary metric |

total tokens、tokens-to-correct、tokens-to-best 不进入 pilot 自动统计；如果需要，只从 Agent/Hook 日志人工读取。

## 6. 显著性检验

### 6.1 基本单位

统计检验不能把 benchmark repeat 当独立样本。正式的 paired unit 应该是：

```text
一个 task x 一个 Agent run
```

如果每个 task 有多个 benchmark shape，应先在 task 内部聚合 shape 结果，例如对多个 shape 的 latency 做 geometric mean，然后形成该 task-run 的一个值。

### 6.2 检验方法

建议同时报告三类结果。

第一，paired bootstrap confidence interval。

对 task 做：每次重采样 task，然后保留该 task 下所有 runs，计算 A3 vs baseline 的 mean log latency difference（也就是 5.2 的内容）。重复若干次，报告 95% CI。

第二，Wilcoxon signed-rank test。（暂时不做）

对每个 task-run 的 paired difference 做 Wilcoxon signed-rank。它不要求差值正态，适合小到中等样本。报告 p-value 和 effect size。

第三，mixed-effects model，作为补充。（暂时不做）

例如：

```text
log_latency ~ arm + category + (1 | task) + (1 | run) + (1 | gpu_id)
```

这个模型可以估计 arm effect，同时控制 task、run、GPU 的随机影响。它不应是唯一证据，但可以作为稳健性分析。

### 6.3 Correctness 的统计

correctness 是 paired binary outcome。建议：

1. A3 vs baseline 的 pass/fail 用 McNemar test；
2. 多组整体比较可用 Cochran's Q；
3. 同时报告 Wilson confidence interval。

### 6.4 Failure Handling

必须提前定义失败 run 如何进入统计。

主表中，以下情况不进入 conditional performance among passers 的性能均值，但必须进入 pass rate 和 failure table：

1. correctness fail；
2. timeout；
3. compile fail。

同时做 conservative sensitivity analysis：

```text
对失败 workspace 赋予惩罚 latency，例如该 task 的 eager latency 或所有 hidden-correct candidates 中最差 latency 的 1.5x；如果没有 hidden-correct candidate，则使用该 task 的 eager latency 或预注册惩罚上限，再计算整体 log latency。
```

这样避免只看 passers 导致 survivorship bias。

## 7. KB 自进化闭环实验

当前还没有验证 v0 -> quarantine -> promotion -> v1 的闭环。这个需要单独设计，不能和主实验混用。

建议设置三份任务集：

| 集合 | 用途 | 是否可进入 KB v1 |
| --- | --- | --- |
| Calibration set | 调试 harness、预算、rubric | 不进入正式结果 |
| Evolution set E | `A3(v0)` 运行，产生经验，进入 quarantine/promotion | 可以进入 KB v1 |
| Held-out test set T | 最终比较 `A3(v1)` vs `A3(v0)` | 绝对不能进入 KB v1 |

推荐规模：

```text
Evolution set: 12 tasks x 3 runs, only A3(v0)
Held-out test set: 36 tasks x 3 runs, A3(v0) vs A3(v1)
```

如果主实验已经使用 36 个正式任务，更严谨的做法是：先冻结 T，不用 T 的任何结果进行 promotion；v1 只来自 E；然后在 T 上同时跑 `A0`/`A1`/`A2`/`A2m`/`A3(v0)`/`A3(v1)`，或者至少跑 `A3(v0)`/`A3(v1)`。

### 7.1 Promotion 规则

KB v1 的 promotion 规则要提前写清楚：

1. 只 promotion hidden-correct 且人工审查无作弊/illegal 的经验；
2. 必须包含 task pattern、failure mode、optimization principle、applicability condition、anti-pattern；
3. 不能包含 hidden shape、具体 benchmark 数字依赖、任务专属 hardcode；
4. 至少一名人工 reviewer 审核；
5. 每条 KB entry 有 provenance、source workspace、applicable operator category、validity constraints；
6. 所有 promoted entries 版本化，形成 KB v1 snapshot；
7. v1 freeze 后才能开始 held-out evaluation。

### 7.2 A3(v1) vs A3(v0) 主指标

建议主指标：

1. hidden correctness pass rate；
2. oracle_best_p50；
3. time-to-correct；
4. negative transfer rate：v1 明显差于 v0 的 task-run 比例。

如果 v1 平均更好但 negative transfer 很高，也要如实报告。这说明自进化需要更严格的 applicability condition 或 retrieval gating。
