# 从 H20 迁移到 AMD、华为等后端

## 1. 保持不变的部分

迁移时不变的是 ECC-KB 方法本体：

- Evidence Capsule schema；
- OpSpec / BackendSpec / ContextPacket 协议；
- 生成、编译、正确性、性能、autotune、复盘的阶段化召回；
- quarantine -> candidate -> stable -> stale -> rejected 状态机；
- compile/correctness/performance/anti-cheating/promotion 门控；
- token/time/iteration/performance/knowledge-utility 指标体系；
- failure signature 与 optimization motif 的表达方式。

## 2. 需要替换的部分

迁移到 AMD 或华为时，需要替换 backend instance layer：

| 组件 | H20 实例 | AMD 实例 | 华为/Ascend 实例 |
|---|---|---|---|
| 语言/DSL | Triton CUDA backend | Triton ROCm / HIP / CK / TileLang | Triton-Ascend / AscendC / CANN / TileLang-like DSL |
| 编译器 | Triton + CUDA toolkit | ROCm clang / HIP / Triton ROCm | CANN / AscendC compiler / Triton-Ascend |
| profiler | torch event, NCU optional | rocprof / Omniperf / Triton metrics | msProf / CANN profiling |
| runtime fingerprint | CUDA driver, SM, Triton | ROCm version, gfx arch, wave size | CANN version, Ascend chip, AI Core/UB features |
| 合法性知识 | CUDA/Triton API constraints | HIP/Triton ROCm constraints | AscendC/Triton-Ascend API, UB/L1/L0 constraints |
| 初始样例 | Triton kernels | ROCm/HIP/Triton examples | CANN/AscendC/Triton-Ascend official examples |
| benchmark runner | PyTorch CUDA | PyTorch ROCm or vendor runtime | MindSpore/PyTorch-Ascend/CANN runner |

## 3. 抽象层如何复用

一个 H20 上得到的 softmax capsule 可以拆成两部分：

```text
portable:
  row-wise softmax must preserve max-subtraction numerical stability;
  mask tail elements for non-power-of-two hidden sizes;
  profile after correctness only;
  avoid hardcoding visible N.
backend_instance_h20:
  Triton block_size = next_power_of_two(N);
  num_warps candidates = [4, 8] for selected H20 shapes;
  measured p50/p95 evidence.
```

迁移到 AMD 或 Ascend 时，portable 部分保留。backend instance 部分必须重新验证、重新 tuning、重新 profile，不能直接复用 H20 的 knob 值。

## 4. 冷启动策略

迁移新后端时，按以下顺序构建最小知识：

1. 后端合法性库：API、dtype、memory scope、同步、资源限制、编译命令、错误码。
2. OpSpec 和 reference harness：先保证正确性测试可运行。
3. 官方样例和已知正确 skeleton：先从 kernel-to-kernel 或 high-level IR 优化开始，不直接从 PyTorch 裸生成低级代码。
4. Profile symptom translator：把 msProf/rocprof/NCU 等后端指标映射到 ECC-KB 的抽象 symptom taxonomy。
5. Tuning record DB：对少量算子族做真实测量，形成 backend-specific warm start。
6. Evolution gates：确保新后端上的成功/失败只先进入 quarantine。

## 5. 如何避免退化成 CUDA 知识库

- stable KB 中的通用 capsule 不允许出现 CUDA-only 字段；
- 所有 CUDA/Triton-H20 证据必须放在 `backend_instances`；
- 检索时先返回 portable reasoning，再附带当前后端实例证据；
- H20 上的 knob 不得作为 AMD/Ascend 默认建议，只可作为“相似机制参考”；
- 每个 promoted capsule 必须标注 `transfer_policy`: `portable`, `requires_revalidation`, `backend_specific`, or `do_not_transfer`。

## 6. H20 结果如何支持方法通用性

H20 MVP 能证明的是机制通用性，而不是 H20 代码通用性。如果 H20 实验显示：

- 同一 schema 能表达语义、性能、失败、门控和演化；
- ContextPacket 比普通 RAG 更能减少无效迭代；
- promoted capsule 能在 held-out shape/dtype 上复用；
- backend-specific evidence 与 portable knowledge 能分离；

则说明 ECC-KB 的组织与演化机制值得迁移。迁移到 AMD/Ascend 仍必须重新采集工具链约束和真机证据。
