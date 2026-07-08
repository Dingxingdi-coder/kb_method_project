# H20 正式实验方案

核心原则：不要只验证 "A3 平均更快"，而要验证三个更强的问题：

- A3 是否在等预算下稳定优于普通 RAG；
- A3 的收益是否来自 ECC-KB 的结构化机制，而不是偶然样本或上下文长度；
- KB 自进化 v1 是否能在严格 held-out 任务上继续提升。

## 0. 预注册 Claims

正式实验应提前写清楚最终要证明的结论。建议拆成四个 claim。

### 0.1 有效性 Claim

在相同 Agent workflow、相同 tuning budget（例如 wall time、candidate 数量、compile 次数等）、相同 H20 harness 下，`A3_ecc_kb` 相比 `A0`/`A1`/`A2` 有更好的 legal hidden-correct 性能。

主指标建议使用：

- best legal p50 latency；
- 或 speedup。

### 0.2 上下文效率 Claim

A3 不是靠塞更多 context 获得效果，而是在 retrieved context tokens 更少或相同上限下取得更好结果。

### 0.3 成本 Claim

A3 的 agent wall time 在 pilot 中更高，因此正式实验不能只报告性能。必须报告：

- time-to-correct；
- time-to-best；
- tokens-to-correct；
- tokens-to-best；
- any-time 曲线。

定义：

- 正确 candidate：从 Agent 开始处理任务，到第一次生成一个 legal + correctness pass 的实现所用时间。
- 最佳 candidate：在固定 budget 内，所有 legal + hidden-correct candidates 中，最终 evaluator 重新 benchmark 后 p50 最低的 candidate。

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

## 3. Budget 和环境控制

### 3.1 Wall Time

pilot 中 A3 平均 wall time 最高，这是正式实验的关键风险。正式实验不能让 A3 用更多时间换性能后只报 p50。

建议设置统一 end-to-end hard cap：

```text
每个 workspace 最大 wall time = 480s
```

此外必须从日志中重构 any-time 指标：

| 截止点 | 指标 |
| --- | --- |
| 120s | 当前 best legal correct candidate 的 p50 / speedup |
| 240s | 当前 best legal correct candidate 的 p50 / speedup |
| 480s | 当前 best legal correct candidate 的 p50 / speedup |

这样可以回答：A3 是更慢但最终更好，还是在同等时间内也更好。

### 3.2 Tuning Budget

正式实验应同时限制 wall time 和 harness 调用次数。

| 预算项 | 建议 |
| --- | --- |
| max modification numbers | 12-16 |
| max compile attempts | 16-24 |
| max correctness runs | 16-24 |
| max benchmark runs | 8-12 |
| max wall time | 600s，所有组相同 |

原因：如果只限制 wall time，不同组可能通过不同数量的 benchmark calls 获得不同程度的 tuning advantage。best p50 尤其容易受 benchmark 调用次数影响。

正式性能建议使用两个版本：

1. `agent_final_legal_p50`：Agent 最终提交的 legal hidden-correct candidate 的 p50；
2. `oracle_best_legal_p50`：在固定预算内生成过的所有 legal hidden-correct candidates 中 p50 最低者。

前者衡量 Agent 最终选择能力，后者衡量生成/搜索能力。主指标可以用 `oracle_best_legal_p50`，但必须同时报告 final，避免过度美化。

### 3.3 Context Budget

所有检索组使用同一最大上限。

| 组别 | retrieved context token cap |
| --- | --- |
| A1 | 12k |
| A2 | 12k |
| A3 | 12k 总上限，分 stage cap |

正式报告中要区分：

1. retrieved context tokens；
2. total tokens；

pilot 里只看 retrieved context tokens 不够。A3 虽然 retrieved tokens 少，但 wall time 高，可能是因为多阶段检索、更多推理回合或更复杂 planning。正式实验必须把 token 和 wall time 分开。

### 3.4 GPU 分配和 Benchmark 控制

H20 benchmark 容易受 GPU 共享、温度、clock、进程干扰影响。建议：

1. 每个 workspace 独占一张 H20，不允许多个 Agent 同时在同一张 GPU 上跑 benchmark。
2. 禁用 MPS/MIG 共享，或者至少保证 benchmark 阶段独占。
3. 记录 GPU ID、driver、CUDA、PyTorch/Triton 版本、power cap、clock policy、温度区间。
4. 同一个 task-run 的不同实验组应尽量在同一类 GPU、同一环境、相近时间窗口内完成。
5. 实验顺序要 randomize/block，不能总是 A0 先跑、A3 后跑。
6. 最终 benchmark 最好由独立 evaluator 统一重跑，而不是直接使用 Agent 自己日志里的最佳数字。
7. 对每个 final/best candidate 做 warmup 后重复 benchmark，报告 p50，同时保存 p20/p50/p80 或 p50/p90。

正式 benchmark 推荐流程：

```text
Agent 运行期间可以调用 correctness 和 benchmark 进行 tuning；这些调用计入预算。
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

即使它 hidden correctness 通过、benchmark 很快，也不能计入正式 best legal p50。

处理方式：

1. 如果 workspace 里还有其他 legal hidden-correct candidate，则 official best 使用最好的 legal candidate。
2. 如果 final candidate illegal，但之前有 legal candidate，final metric 记 final illegal，同时可报告 best legal。
3. 如果整个 workspace 没有 legal hidden-correct candidate，则该 workspace 记为 legal correctness fail。
4. cheating/illegal attempt rate 单独报告。

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
6. 最终报告 cheating rate、illegal-final rate、illegal-best rate。

## 5. 统计指标

正式指标分为 correctness、performance、cost、context efficiency、legality 五类。

### 5.1 Correctness 指标

| 指标 | 定义 |
| --- | --- |
| public correctness pass rate | Agent 可见 correctness 是否通过 |
| hidden correctness pass rate | hidden correctness 是否通过 |
| legal hidden correctness pass rate | 同时 hidden correct 且 legality pass |
| first-correct rate | 在预算内是否至少生成过一个 legal correct candidate |
| final-correct rate | 最终提交 candidate 是否 legal correct |
| correctness failure reason | compile fail / runtime fail / numerical mismatch / timeout / illegal API |

主表应使用 legal hidden correctness pass rate，不是普通 hidden correctness pass rate。

### 5.2 Performance 指标

| 指标 | 定义 |
| --- | --- |
| final legal p50 | Agent 最终提交的 legal hidden-correct candidate 的 p50 |
| oracle best legal p50 | 固定预算内所有 legal hidden-correct candidates 中 p50 最低者 |
| speedup vs eager | eager p50 / candidate p50 |
| speedup vs torch.compile | torch.compile p50 / candidate p50 |
| per-task win rate | 某组在同一 task-run 上是否优于对照组 |
| category-level p50 / speedup | 按算子类别聚合 |
| p90 或 tail latency | secondary metric |

聚合时不建议直接用 arithmetic mean latency 或 arithmetic mean speedup。正式报告应使用：

1. 对 latency 用 log latency 聚合；
2. 对 speedup 用 geometric mean；
3. 对每个 task 先在 shapes 内聚合，再在 runs 和 tasks 上聚合；
4. 同时报告 median paired improvement。

推荐主性能指标：

```text
paired log latency improvement of A3 vs A2 on oracle_best_legal_p50
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
| GPU active time | harness/correctness/benchmark 占用 GPU 的时间 |
| retrieval wall time | 检索耗时 |
| LLM wall time | LLM 调用耗时 |
| compile time | 编译耗时 |
| benchmark time | Agent tuning benchmark 耗时 |
| number of candidates | 生成 candidate 数 |
| number of benchmark calls | tuning benchmark 调用数 |
| number of correctness calls | correctness 调用数 |

A3 pilot 的主要短板是 wall time，所以正式实验必须做：

1. average / median wall time；
2. p90 wall time；
3. timeout rate；
4. time-to-first-legal-correct；
5. time-to-best；
6. any-time best p50 at 150s / 300s / 600s。

### 5.4 Token 和 Context Efficiency 指标

| 指标 | 定义 |
| --- | --- |
| retrieved context tokens | 检索返回内容 tokens |
| total tokens | input + completion |
| speedup per 1k retrieved tokens | context efficiency secondary metric |
| latency reduction per 1k total tokens | cost-normalized secondary metric |

pilot 的 retrieved context tokens 更少是有用信号，但正式实验还要看 total tokens。

### 5.5 Legality / Cheating 指标

| 指标 | 定义 |
| --- | --- |
| illegal candidate rate | 所有 generated candidates 中 illegal 比例 |
| illegal final rate | final candidate illegal 比例 |
| illegal oracle-best-before-filter rate | 未过滤 best candidate 中 illegal 比例 |
| high-level fallback rate | 使用 target high-level API fallback 的比例 |
| harness tampering rate | 修改或规避 harness 的比例 |
| shape-specialization violation rate | 非法利用 public benchmark shape 或 hidden leak 的比例 |

这类指标要按实验组报告。A3 如果不仅性能好，而且 illegal rate 更低，也可以成为 ECC-KB 的额外价值点：结构化知识可能减少错误路径。

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

对 task 做 cluster bootstrap：每次重采样 task，然后保留该 task 下所有 runs，计算 A3 vs baseline 的 mean log latency difference。重复 10,000 次，报告 95% CI。

第二，Wilcoxon signed-rank test。

对每个 task-run 的 paired difference 做 Wilcoxon signed-rank。它不要求差值正态，适合小到中等样本。报告 p-value 和 effect size。

第三，mixed-effects model，作为补充。

例如：

```text
log_latency ~ arm + category + (1 | task) + (1 | run) + (1 | gpu_id)
```

这个模型可以估计 arm effect，同时控制 task、run、GPU 的随机影响。它不应是唯一证据，但可以作为稳健性分析。

### 6.3 Correctness 和 Legality 的统计

correctness / legality 是 paired binary outcome。建议：

1. A3 vs baseline 的 pass/fail 用 McNemar test；
2. 多组整体比较可用 Cochran's Q；
3. 同时报告 Wilson confidence interval；
4. 对 illegal rate 单独做 paired 或 bootstrap 分析。

### 6.4 Failure Handling

必须提前定义失败 run 如何进入统计。

主表中，以下情况不进入 conditional performance among passers 的性能均值，但必须进入 pass rate 和 failure table：

1. correctness fail；
2. illegal final；
3. timeout；
4. compile fail。

同时做 conservative sensitivity analysis：

```text
对失败 workspace 赋予惩罚 latency，例如该 task 的 eager latency 或所有 legal candidates 中最差 latency 的 1.5x，再计算整体 log latency。
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

1. 只 promotion legal hidden-correct 的经验；
2. 必须包含 task pattern、failure mode、optimization principle、applicability condition、anti-pattern；
3. 不能包含 hidden shape、具体 benchmark 数字依赖、任务专属 hardcode；
4. 至少一名人工 reviewer 审核；
5. 每条 KB entry 有 provenance、source workspace、applicable operator category、validity constraints；
6. 所有 promoted entries 版本化，形成 KB v1 snapshot；
7. v1 freeze 后才能开始 held-out evaluation。

### 7.2 A3(v1) vs A3(v0) 主指标

建议主指标：

1. legal hidden correctness pass rate；
2. oracle_best_legal_p50；
3. time-to-correct；
4. tokens-to-best；
5. illegal candidate rate；
6. negative transfer rate：v1 明显差于 v0 的 task-run 比例。

如果 v1 平均更好但 negative transfer 很高，也要如实报告。这说明自进化需要更严格的 applicability condition 或 retrieval gating。
