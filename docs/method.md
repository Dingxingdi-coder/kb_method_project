# ECC-KB 方法说明：可执行证据胶囊知识库

## 1. 方法摘要

ECC-KB（Executable Evidence-Capsule Knowledge Base）面向自动算子生成与优化。它的基本判断是：自动算子生成系统真正缺少的不是更多文档，而是可被执行控制面使用、可被实验更新、可被验证门控约束的知识。知识库应当向任意 Coding Agent 输出阶段化、低噪声、可执行、可验证的上下文，并在每次真实编译、运行、正确性测试和 profiling 后更新。

ECC-KB 的知识单元称为 **Evidence Capsule**。一个 Evidence Capsule 是一个条件化动作单元：

```text
when(task_state, op_contract, backend_abstract_features, evidence_scope)
  -> suggest(action | transformation | guard | stop)
  -> expect(effect_on_correctness | effect_on_latency | effect_on_search_space)
  -> verify(gates)
  -> invalidate(failure_boundary)
```

它不是普通文档 chunk。它必须说明：什么时候适用，为什么适用，如何执行，如何验证，什么时候失效，哪些证据支撑，如何从 H20 实例迁移到其他后端。

## 2. 核心设计原则

### 2.1 后端无关优先

ECC-KB 先定义抽象后端模型，再实例化到具体芯片。抽象后端模型只包含可迁移的机制维度：

- execution hierarchy：program / block / warp-or-wave / lane / vector / tensor-core-like unit；
- memory hierarchy：global / cache / scratchpad / register / special on-chip buffer；
- movement pattern：coalesced load、gather/scatter、tile copy、vectorized load、bank-conflict risk；
- synchronization model：block-local sync、warp-local sync、pipeline barrier、async copy dependency；
- numerical model：accumulation dtype、fast math、exp/log approximation、rounding and deterministic behavior；
- compiler/toolchain model：DSL legality、intrinsic support、resource limits、launch overhead、autotune knobs；
- profiler symptom taxonomy：memory-bound、compute-bound、occupancy-limited、register-pressure、bank-conflict、divergence、launch-bound。

H20 只是一个 backend instance。AMD、Ascend、TPU 或其他后端只替换实例层、工具链适配器、profile 解释器和后端约束证据，不替换方法本身。

### 2.2 Coding Agent 无关

ECC-KB 不依赖某个 Agent 的内部记忆或私有工具。Agent 只需要能读写文件、调用 CLI/API 或接收 structured prompt。知识服务接口固定为：

```text
Input:  OpSpec + Phase + CurrentState + BackendInstance + Goal + Budget
Output: ContextPacket = {must_obey, recommended_actions, anti_actions, validation_plan, evidence_refs, stop_conditions}
```

Agent 可以是 Codex、Claude Code、Cursor Agent、OpenHands、自研 agent 或普通脚本。Harness 负责验证、计时、终止、回滚和入库；Agent 只负责提出候选代码或修改。

### 2.3 正确性和性能分层

ECC-KB 明确拆分：

- **Semantic/Correctness capsules**：保护算子语义、shape/dtype/layout、容差、边界条件、anti-cheating、fuzzing。
- **Performance capsules**：在正确候选上缩小优化空间，给出结构性改写、参数性搜索和停止条件。

修复 loop 和性能 loop 不能混在一起。编译/正确性失败时只召回修复相关知识；正确性通过后才召回 profile motif 和 autotune 记录。

### 2.4 证据优先

任何知识进入稳定库前，必须具备可复现证据。自然语言总结、LLM 自评、单次 allclose 或单次 benchmark 只能进入 quarantine，不得直接进入 stable。

## 3. 知识定义

ECC-KB 中的“知识”定义为：

> 一个绑定到算子契约、任务阶段、抽象后端特征和验证证据的条件化工程动作单元。该单元能够减少候选搜索空间、避免已知失败、指导修复或优化，并能通过指定门控被确认、修正、降级或废弃。

该定义适合自动算子生成，因为 kernel 生成的关键不是“知道一句优化建议”，而是知道：

- 这条建议是否适用于当前 op/shape/dtype/layout；
- 它会改变语义还是只改变 schedule；
- 它在当前后端实例是否合法；
- 它预期改善哪个指标；
- 它是否有真机证据；
- 它在哪些边界下会失败；
- 它如何被 Agent 执行；
- 它如何被 harness 验证。

## 4. 知识单元类型

ECC-KB 至少维护七类 Knowledge Unit。

| 类型 | 主要用途 | 典型触发阶段 |
|---|---|---|
| `op_contract` | 算子语义、shape/dtype/layout、容差、reference、隐藏测试规则 | 生成、正确性 |
| `legality_constraint` | DSL/API/intrinsic/resource 合法性约束 | 生成、编译修复 |
| `implementation_skeleton` | 可编辑代码骨架或 IR/schedule 模板 | 生成 |
| `failure_signature` | 编译、数值、越界、race、cheating、性能退化模式 | 修复、复盘 |
| `optimization_motif` | `trigger -> diff/action -> expected profile change -> gates` | 性能优化 |
| `tuning_record` | workload fingerprint、knobs、latency、profile、版本 | autotune、warm start |
| `stop_capsule` | 已接近上界、收益不足、风险过高、重复失败 | 搜索终止 |

这些单元通过统一 schema 表示，但状态和门控不同。

## 5. 双层知识结构

ECC-KB 将知识分成两层。

### 5.1 后端无关层

后端无关层保存：

- 算子语义和验证方法；
- 算子族级别的计算/访存模式；
- 抽象优化 motif，例如“行内 softmax 需要 max-subtract-exp-sum-divide 的两阶段或融合阶段语义”；
- 抽象 profile bottleneck 分类；
- 跨后端可复用的 failure boundary；
- 从 op contract 到测试矩阵的生成规则。

### 5.2 后端实例层

后端实例层保存：

- 具体后端 DSL/API 合法性；
- H20 / AMD / Ascend 的 target fingerprint；
- 工具链版本、编译器行为、profile 计数器映射；
- 具体 tile/wrap/num_warps/stages/vector width/UB size 等 knobs；
- 真实 benchmark 与 profile 证据；
- 后端特化的代码模板或 kernel 片段。

两层通过 `backend_abstract_features` 和 `backend_instances` 字段连接。一个通用 capsule 可以有多个实例证据；一个实例证据不得反向污染通用规则，除非通过跨 shape 或跨后端证据提升。

## 6. 知识抽取机制

ECC-KB 从六类来源抽取知识。

### 6.1 调研报告与公开资料

抽取对象不是段落摘要，而是“可验证命题”。例如：

```text
source_observation:
  自动生成算子候选需要经过真实编译、真机运行、correctness 检查和 profiler 反馈。
extracted_capsule:
  type: validation_policy
  phase: all
  condition: candidate_kernel_created
  action: run compile -> correctness -> benchmark -> profile, in this order
  gate: no performance capsule can be promoted before correctness gate passes
```

### 6.2 官方文档和源码

抽取 API 合法性、dtype 支持、layout 限制、intrinsic 约束、编译 flag、profile counter 语义。此类知识进入 `legality_constraint` 或 `backend_instance`，默认只在对应版本有效。

### 6.3 已有 kernel 和 expert implementation

抽取 skeleton、schedule trace、关键边界处理、mask 策略、accumulation dtype、输入假设。代码本身不是最终知识，必须经过 contract 对齐和 evidence 绑定。

### 6.4 Benchmark 和 profile 结果

抽取 tuning record、profile symptom、stop capsule 和 optimization motif。单次 benchmark 只能形成候选证据；至少重复测量、p50/p95 稳定、正确性通过后才能形成 promoted performance capsule。

### 6.5 Agent 轨迹

抽取 action diff、失败原因、修复路径、被证伪假设和有效 motif。禁止把完整对话直接当知识。轨迹需要压缩为：

```text
state_signature -> attempted_action -> result -> diagnosis -> reusable_boundary
```

### 6.6 反向退优化与 ablation

对已成功 kernel，可反向移除或替换某个优化，观察性能/正确性变化。只有存在 `bad -> good` 或 `good -> ablated_bad` 的差分证据，才把经验升级为因果 motif。

## 7. 知识构建流程

ECC-KB 的构建流程如下。

```text
raw materials
  -> source registry
  -> claim extraction
  -> canonical OpSpec / BackendSpec alignment
  -> capsule drafting
  -> static schema validation
  -> quarantine
  -> evidence collection
  -> promotion gate
  -> stable KB
  -> retrieval index refresh
```

### 7.1 识别可迁移知识

判断一条知识是否可迁移，不看它是否含有 CUDA/Triton/AscendC 关键词，而看它是否能重写成抽象后端特征。例如：

- “block 内 reduction 先局部聚合再跨 block 聚合”可迁移；
- “`num_warps=8` 在某个 H20 shape 上最快”不可直接迁移，只是实例证据；
- “softmax 需要 max-subtract 防溢出”属于语义与数值稳定知识，可迁移；
- “Triton `tl.arange` 需要 power-of-two block 某些写法更易编译”属于 DSL/backend instance 知识。

### 7.2 识别后端特化知识

后端特化知识满足以下任一条件：依赖具体 API、具体资源上限、具体 profiler counter、具体 compiler bug、具体 shape/driver/toolchain、具体 memory scope 名称。它必须带 target fingerprint。

### 7.3 把经验变成可验证知识

经验必须被结构化为：

```text
Observation: 在 task X 上做 Y 后 latency 下降。
Capsule: 当 condition C 成立时，可尝试 action Y；预期影响 metric M；需通过 gate G；在 boundary B 下失效。
Evidence: run_id, code_hash_before, code_hash_after, workload_key, measurements, profile_delta, correctness_matrix.
```

### 7.4 处理冲突知识

冲突知识不直接合并。ECC-KB 采用四级冲突处理：

1. 若 target fingerprint 不同，保留为后端实例差异。
2. 若 shape/dtype/layout 不同，收窄条件边界。
3. 若同条件下证据冲突，降级到 quarantine 并要求重测。
4. 若长期证据显示失效，标记 `stale`，保留为负样本。

### 7.5 证据绑定

每个 promoted capsule 必须至少绑定：

- source refs；
- op contract hash；
- code diff hash 或 schedule trace；
- environment fingerprint；
- correctness test matrix；
- performance measurement distribution；
- decision lineage；
- promotion gate result。

## 8. 检索与使用机制

ECC-KB 不返回“相关文章”。它返回 ContextPacket。

### 8.1 ContextPacket 格式

```yaml
phase: performance_optimize
must_obey:
  - capsule_id: sem_softmax_stability_v1
    instruction: preserve max-subtract-exp-sum-divide semantics; do not approximate by exp(x)/sum(exp(x)) without max subtraction
recommended_actions:
  - capsule_id: perf_row_softmax_tile_v2
    action: try one program per row for N <= 4096; tune block size to next_power_of_two(N)
    expected_effect: reduce global memory passes and improve p50 latency
anti_actions:
  - capsule_id: fail_hardcode_shape_v1
    instruction: do not specialize to visible shapes unless OpSpec declares fixed shape
validation_plan:
  - run hidden shape sweep
  - run dtype fp16/bf16 tolerance matrix
  - benchmark p50/p95 with 100 warmup + 500 repeats
stop_conditions:
  - if speedup < 1.05 after 3 parameter attempts and profile shows launch-bound, stop local tuning
```

### 8.2 阶段化召回

| 阶段 | 召回内容 | 不召回内容 |
|---|---|---|
| plan/generate | OpSpec、合法模板、同类 skeleton、禁止行为 | 低层 profile knobs 全量历史 |
| compile repair | error signature、API 合法性、最小修复 diff | 性能优化 motif |
| correctness repair | hidden edge cases、容差、fuzz seed 类别、anti-cheating | profiler counter |
| performance optimize | profile symptom -> motif、tuning record、stop capsule | 大段官方文档 |
| autotune | knob search space、历史最优、失效组合 | 语义长解释 |
| retrospective | trajectory compression、promotion/reject 建议 | 未验证的成功断言 |

### 8.3 降噪策略

ECC-KB 先硬过滤，再相似召回，再证据重排。硬过滤包含 op family、phase、dtype、layout、shape bucket、backend abstract feature、target fingerprint 和 toolchain version。相似召回只在硬过滤后的候选中进行。重排优先级为：合法性 > 证据强度 > 条件匹配 > 近期复现 > 语义相似 > 文本相关。

简化伪代码：

```python
def retrieve_context(query, kb):
    candidates = kb.lookup_by_phase(query.phase)
    candidates = hard_filter(candidates, query.op_contract, query.backend_instance)
    candidates += structural_neighbors(query.op_fingerprint, query.shape_bucket)
    candidates += semantic_neighbors(query.error_or_profile_summary)
    candidates = [c for c in dedup(candidates) if legality_pass(c, query)]
    ranked = rank_by_evidence_and_goal(candidates, query.goal)
    return build_context_packet(ranked[:query.topk], query.phase)
```

## 9. 自我进化机制

ECC-KB 的自我进化不是把日志越存越多，而是把实验轨迹转化为可复用 capsule。

### 9.1 每次实验记录

每个 run 记录：

- task/op spec；
- backend fingerprint；
- agent identity as external label only；
- prompt/context packet hash；
- code diff and candidate artifact hash；
- compile result；
- correctness matrix；
- benchmark distribution；
- profile summary；
- decision: KEEP / DISCARD / FAIL / NEED_MORE_EVIDENCE；
- token/time/GPU cost；
- parent capsule ids used。

### 9.2 成功经验写回

成功经验先写入 quarantine。只有满足：正确性强门控、至少多次 benchmark、无 p95 退化、profile 归因清晰、条件边界明确、对至少一个 ablation 或 parent comparison 有证据，才能生成 promoted motif。

### 9.3 失败经验写回

失败经验更容易写入 failures，但不能变成禁止规则，除非在相同条件下重复出现。失败记录按 signature 聚类：compile error、runtime error、numerical mismatch、mask/overrun、race/nondeterminism、cheating/fallback、performance regression、timeout/resource error。

### 9.4 知识状态机

```text
draft -> quarantine -> candidate -> stable -> stale -> rejected
                      \-> need_more_evidence
```

- `draft`：人工或 LLM 草拟，未校验。
- `quarantine`：schema 合法，有来源，但未通过实验门控。
- `candidate`：通过部分门控，可低权重召回。
- `stable`：通过 promotion gate，可默认召回。
- `stale`：曾有效，但当前环境或新证据下失效。
- `rejected`：被证伪，不再作为建议，只作为负样本。

### 9.5 防污染门控

进入 stable 前必须通过：

1. schema gate：字段完整，条件和边界明确；
2. static gate：DSL/API/target/version 合法；
3. compile gate：候选独立编译成功；
4. correctness gate：smoke、shape sweep、dtype matrix、edge cases、hidden seeds；
5. anti-cheating gate：不调用 reference/fallback，不 hardcode visible shapes，不跳过计算；
6. performance gate：重复计时，p50/p95，noise control，baseline 明确；
7. attribution gate：profile delta 或 ablation 支持预期机制；
8. reproducibility gate：环境指纹、代码 hash、数据 seed 和命令可复现。

## 10. 为什么不是普通 RAG / KG / 规则库

ECC-KB 吸收检索、图关系、规则、模板、benchmark harness 的有用部分，但主贡献不在这些组件本身，而在 **知识单元的可执行证据化与演化门控**。

普通 RAG 返回文本；ECC-KB 返回阶段化 action packet。普通知识图谱连接实体；ECC-KB 的边必须能参与合法性过滤、执行建议、证据重排和失效判定。规则库保存经验；ECC-KB 要求规则携带反例、验证协议和证据 lineage。Benchmark loop 只测试候选；ECC-KB 把测试结果蒸馏成可迁移 capsule。Agent memory 只记录过去；ECC-KB 用 quarantine/promotion/stale/reject 状态机决定哪些过去能影响未来。

## 11. 预期因果链

ECC-KB 通过以下路径降低 time/token/iteration：

1. OpSpec 和 must_obey capsule 避免语义误解和 hardcode visible shapes。
2. Legality capsule 在生成前过滤非法 API、dtype、layout 和 resource 组合，减少编译失败。
3. Failure signature capsule 让 Agent 看到结构化根因，而不是长编译日志。
4. Correctness capsule 把隐藏边界和容差提前暴露，避免靠少量 allclose 误判。
5. Profile motif capsule 只在正确后给出少量结构性/参数性建议，避免把全量 profiler 输出塞进上下文。
6. Tuning record 和 stop capsule 缩小搜索空间，避免重复尝试低收益 knobs。
7. Evolution gate 只让被验证知识进入 stable，避免错误经验累积导致后续任务更差。

## 12. 最小实现建议

第一版不需要训练模型，也不需要多后端硬件。实现以下最小组件即可：

```text
schemas/
  knowledge_unit.schema.json
  evolution_record.schema.json
kb/
  stable/*.json
  quarantine/*.json
  failures/*.json
  evidence/*.jsonl
scripts or tools/
  validate_schema
  retrieve_context
  ingest_run
  promote_capsule
  export_context_packet
experiments/h20/
  fixed operator tasks
  fixed baselines
  fixed metrics
```

检索第一版可以只用 JSON + SQLite + BM25/简单向量，关键不是索引技术，而是 capsule schema、阶段化 context packet 和 promotion gate。
