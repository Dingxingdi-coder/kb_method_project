# Sources: 初始资料库

`sources/` 保存用于构建 H20 MVP 初始资料库的源材料索引和派生记录。首轮实验必须先冻结 `raw_corpus_v0`，再从同一批资料生成 `KB v0`。

## 目标

资料库用于支持四组主实验：

- `A1_raw_corpus_rag` 直接对 `raw_corpus_v0` 做普通 RAG；
- `A2_kb_plain_rag` 对由 `raw_corpus_v0` 转换出的 `KB v0` 做普通 RAG；
- `A3_ecc_kb` 使用由同一 `KB v0` 构建的 ECC-KB ContextPacket；
- `A0_prompt` 不读取资料库或知识库。

这个设计保证 A1、A2、A3 的资料来源一致。实验差异来自资料形态和使用方法，而不是资料覆盖范围。

## 建议目录

```text
sources/
  README.md
  registry/
    sources.yaml              # 每条资料的 source_id、URL/path、license、timestamp、topic、backend tags
  raw_corpus/
    <source_id>.md            # 清洗后的原始资料 chunk，保持来源可追溯
  derived_claims/
    <source_id>.claims.yaml   # 从原始资料抽取出的可验证 claim
```

Git 不追踪空目录。实际填充资料前，可以先只保留本说明文件。

## Source registry 字段

每条资料至少记录以下字段：

```yaml
source_id: triton_docs_programming_model_2026_07
kind: official_doc | paper | report | code | benchmark | blog | tool_doc
uri: https://example.com/path
local_path: sources/raw_corpus/triton_docs_programming_model_2026_07.md
license: unknown | permissive | cc_by | apache_2 | mit | other
ingested_at: 2026-07-05
topics:
  - triton
  - programming_model
  - legality
backends:
  - nvidia_cuda
  - amd_rocm
trust_level: official | peer_reviewed | project_report | community | unverified
usable_for:
  - raw_corpus_rag
  - claim_extraction
notes: short audit note
```

## 冻结规则

`raw_corpus_v0` 冻结后，不允许在同一轮实验中继续加入资料。若必须补资料，创建 `raw_corpus_v1`，并把使用 `v1` 的 run 与使用 `v0` 的 run 分开比较。

`KB v0` 必须只从 `raw_corpus_v0` 派生。A2 和 A3 不能看到 A1 不可见的额外资料。