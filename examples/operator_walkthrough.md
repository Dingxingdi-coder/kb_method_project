# 贯穿样例：Row Softmax

该样例说明 ECC-KB 如何从原始材料抽取知识、组织知识、服务 Agent，并在实验后写回。样例不表示方法绑定 softmax。

## 1. OpSpec

```yaml
op_family: softmax
signature: y = softmax(x, dim=-1)
inputs:
  x: {shape: [B, N], dtype: [fp16, bf16, fp32], layout: contiguous_or_strided}
outputs:
  y: {shape: [B, N], dtype: same_as_input}
semantics:
  y[i,j] = exp(x[i,j] - max_j x[i,j]) / sum_j exp(x[i,j] - max_j x[i,j])
constraints:
  preserve numerical stability
  support non-power-of-two N by masking
  do not hardcode visible shapes
correctness:
  reference: torch.softmax(x, dim=-1)
  tolerance: dtype dependent
```

## 2. 从材料抽取的通用 capsule

```yaml
id: sem_softmax_stability_v1
type: op_contract
abstraction_level: portable
phase: [generate, correctness_repair]
valid_when:
  op_family: softmax
  axis: last_dim
action:
  must: subtract row max before exp
  must: mask tail elements before max/sum for padded block
expected_effect:
  correctness: prevents overflow and masked-tail pollution
validation:
  include large positive/negative values
  include non-power-of-two N
invalid_when:
  alternative mathematically equivalent implementation is proven and passes hidden tests
```

## 3. H20 实例化 capsule

```yaml
id: perf_h20_triton_row_softmax_tile_v1
type: optimization_motif
abstraction_level: backend_instance
phase: performance_optimize
parent: sem_softmax_stability_v1
backend_instances:
  - vendor: nvidia
    device_class: h20_runtime_fingerprint
    dsl: triton
valid_when:
  op_family: softmax
  shape_pattern: [B, N]
  N_range: [128, 4096]
action:
  try one Triton program per row
  use BLOCK_N = next_power_of_two(N)
  tune num_warps in candidate set declared by backend adapter
expected_effect:
  reduce global memory passes for medium N
validation:
  hidden correctness first
  benchmark p50/p95 after correctness
status: quarantine_until_measured
```

## 4. Agent 使用

生成阶段，ECC-KB 返回：

```yaml
must_obey:
  - preserve max-subtract-exp-sum-divide semantics
  - mask offsets >= N
recommended_actions:
  - start from row-wise Triton skeleton
anti_actions:
  - do not specialize to N=1024 only
validation_plan:
  - run quick correctness on N=128,257,1024,4096
```

性能阶段，若 correctness 已通过，ECC-KB 返回：

```yaml
recommended_actions:
  - tune BLOCK_N and num_warps within backend adapter range
  - inspect memory-bound vs occupancy-limited symptom
stop_conditions:
  - stop local tuning if three valid attempts produce <1.05 speedup and profile is launch-bound
```

## 5. 写回

若 Agent 的候选在 `[1024,257]`、`[4096,1024]`、`[512,4096]` 上通过 hidden correctness，并且 p50 相对 `torch.compile` 改善，harness 生成 evolution record。若 profile 归因支持 memory pass 降低或 occupancy 改善，`perf_h20_triton_row_softmax_tile_v1` 可从 quarantine 升级到 candidate 或 stable。

若某个尝试把 `N` 写死成 1024，在公开测试通过但 hidden N=257 失败，则写入 failure capsule：

```yaml
id: fail_softmax_hardcode_visible_n_v1
type: failure_signature
phase: correctness_repair
signature: visible_shape_hardcode
condition:
  op_family: softmax
  generated_code_contains: constants_matching_public_N
action:
  reject candidate and require shape-parametric indexing
status: stable_negative_after_reproduced
```

## 6. 后端无关与后端实例分离

portable knowledge：softmax 数值稳定、tail mask、hidden shape sweep、anti-hardcode。

H20 instance knowledge：Triton skeleton、BLOCK_N/num_warps 搜索范围、H20 测量分布、NCU/Triton profile symptom。

迁移到 AMD/Ascend 时，portable capsule 保留；H20 instance capsule 只能作为 `requires_revalidation` 参考，不得直接 promoted 到新后端。
