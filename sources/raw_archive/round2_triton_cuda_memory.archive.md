---
archive_id: round2_triton_cuda_memory
source_ids:
  - triton_pointwise_tutorial_round2
  - cuda_memory_access_round2
retrieved_at: 2026-07-11
archive_kind: concise_locator_and_paraphrase
license: mixed
---

# Triton pointwise and CUDA memory-access locators

Official locators:
- Triton vector-add tutorial: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

Source-grounded notes:
- The Triton tutorial maps a program id to a block of contiguous offsets, masks `offsets < n_elements`, and applies the mask to both loads and stores.
- The tutorial validates against PyTorch before comparing providers across sizes. It therefore supports a correctness-first comparison methodology, not a blanket speedup claim.
- CUDA guidance favors adjacent-lane access to adjacent data for coalescing and uses effective bandwidth as a diagnostic derived from bytes moved and kernel time.
- Strided and irregular accesses may fetch memory segments that contain unused data. A regular transpose can use tiling to make global-memory legs coalesced, while gather/scatter behavior depends on the index distribution.

Applicability:
- Contiguous pointwise kernels, regular layout copies, and memory-leg analysis for irregular indexing.

Limitations:
- Concrete block sizes, vector widths, cache behavior, occupancy, and measured bandwidth are architecture- and shape-dependent.
- No statement here is an H20 benchmark result.
