# H20 Expanded Pilot Test Objects

This file freezes the redesigned KernelBench-X aligned operator tasks for the
expanded pilot workspace set. The selected tasks are taken from
`/data/dxd/KernelBenchX/data/kernelbenchx`.

Each row is one experimental task. The KernelBench-X task file is the source of
truth for callable semantics, test cases, tolerances, and post-hoc KBX
evaluation. During H20 runs, `candidate.py` remains the submission container, but
it must expose the listed top-level KernelBench-X entrypoint with compatible
semantics. The post-hoc exporter can then copy the same implementation to the
matching KernelBench-X basename.

Before launching a formal run, regenerate the H20 task JSON/workspaces from this
spec so that the prompt, harness, and post-hoc export manifest agree with these
operator choices.

## Summary

- Task count: 36
- Categories: 6
- Tasks per category: 6
- Selection principle: choose KernelBench-X operators that match the six H20
  kernel-pattern categories and are directly evaluable by the KernelBench-X
  official pipeline.
- Experimental groups: `A0_prompt`, `A1_raw_corpus_vector_rag`,
  `A2_kb_vector_rag`, `A3_ecc_kb`

## Pointwise / Fused Memory-Bound

| task_id | KBX op | KBX task file | dtype/profile | required entrypoint |
|---|---|---|---|---|
| `expanded_pilot_pointwise_add` | `add` | `Math/add.py` | elementwise fp32-style tests | `add(input, other, alpha=1, out=None)` |
| `expanded_pilot_pointwise_mul` | `mul` | `Math/mul.py` | elementwise fp32-style tests | `mul(input, other, out=None)` |
| `expanded_pilot_pointwise_gelu_fp16` | `gelu_fp16` | `Activation/gelu_fp16.py` | fp16 activation | `gelu_fp16(input, approximate='none')` |
| `expanded_pilot_pointwise_gelu_bf16` | `gelu_bf16` | `Activation/gelu_bf16.py` | bf16 activation | `gelu_bf16(input, approximate='none')` |
| `expanded_pilot_pointwise_fused_add_gelu` | `fused_add_gelu` | `Fusion/fused_add_gelu.py` | fused add + GELU | `fused_add_gelu(input, other, alpha=1, approximate='none', out=None)` |
| `expanded_pilot_pointwise_fused_mul_sub` | `fused_mul_sub` | `Fusion/fused_mul_sub.py` | fused multiply + subtract | `fused_mul_sub(input, other_mul, other_sub, alpha=1, out=None)` |

## Reduction

| task_id | KBX op | KBX task file | dtype/profile | required entrypoint |
|---|---|---|---|---|
| `expanded_pilot_reduction_sum` | `sum` | `Reduce/sum.py` | sum over scalar/tuple dims | `sum(input, dim, keepdim=False, dtype=None)` |
| `expanded_pilot_reduction_mean` | `mean` | `Reduce/mean.py` | mean over scalar/tuple dims | `mean(input_tensor, dim, keepdim=False, dtype=None, out=None)` |
| `expanded_pilot_reduction_std` | `std` | `Reduce/std.py` | std over optional dims | `std(input, dim=None, correction=1, keepdim=False, out=None)` |
| `expanded_pilot_reduction_min` | `min` | `Reduce/min.py` | value + index reduction | `min(input_tensor, dim, keepdim=False)` |
| `expanded_pilot_reduction_argmax` | `argmax` | `Reduce/argmax.py` | index reduction | `argmax(input_tensor, dim, keepdim=False)` |
| `expanded_pilot_reduction_fused_sum_std` | `fused_sum_std` | `Fusion/fused_sum_std.py` | sum + std fused reduction | `fused_sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None)` |

## Softmax

| task_id | KBX op | KBX task file | dtype/profile | required entrypoint |
|---|---|---|---|---|
| `expanded_pilot_softmax_softmax` | `softmax` | `Math/softmax.py` | dim 0/1/-1 + optional dtype | `softmax(input, dim, dtype=None)` |
| `expanded_pilot_softmax_fused_softmax_log` | `fused_softmax_log` | `Fusion/fused_softmax_log.py` | log + softmax | `fused_softmax_log(input, dim=-1, dtype=None)` |
| `expanded_pilot_softmax_fused_log_softmax_linear` | `fused_log_softmax_linear` | `Fusion/fused_log_softmax_linear.py` | linear + log_softmax | `fused_log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None)` |
| `expanded_pilot_softmax_fused_repeat_interleave_log_softmax` | `fused_repeat_interleave_log_softmax` | `Fusion/fused_repeat_interleave_log_softmax.py` | repeat_interleave + log_softmax | `fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None)` |
| `expanded_pilot_softmax_fused_cross_entropy_log_softmax` | `fused_cross_entropy_log_softmax` | `Fusion/fused_cross_entropy_log_softmax.py` | cross entropy + log_softmax | `fused_cross_entropy_log_softmax(input, target, dim=1, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)` |
| `expanded_pilot_softmax_attention` | `attention` | `Fusion/attention.py` | scaled dot-product attention | `attention(q, k, v, causal=False, softmax_scale=None, *, out=None)` |

## Normalization / LayerNorm-Like

KernelBench-X has few pure fp16/bf16 layernorm-forward tasks. This category uses
the closest directly evaluable KBX normalization and layernorm-like tasks.

| task_id | KBX op | KBX task file | dtype/profile | required entrypoint |
|---|---|---|---|---|
| `expanded_pilot_norm_layernorm_w8a8` | `layernorm_w8a8` | `Quantization/layernorm_w8a8.py` | dynamic W8A8 layernorm | `layernorm_w8a8(input, normalized_shape, weight=None, bias=None, eps=1e-5)` |
| `expanded_pilot_norm_fused_layer_norm_relu_linear` | `fused_layer_norm_relu_linear` | `Fusion/fused_layer_norm_relu_linear.py` | linear + ReLU + layer_norm | `fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True)` |
| `expanded_pilot_norm_fused_cross_entropy_softmax_layernorm` | `fused_cross_entropy_softmax_layernorm` | `Fusion/fused_cross_entropy_softmax_layernorm.py` | CE + softmax + layernorm | `fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None)` |
| `expanded_pilot_norm_fused_silu_layer_norm_conv2d` | `fused_silu_layer_norm_conv2d` | `Fusion/fused_silu_layer_norm_conv2d.py` | conv2d + layer_norm + SiLU | `fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5)` |
| `expanded_pilot_norm_fused_bmm_rmsnorm_gelu_dropout` | `fused_bmm_rmsnorm_gelu_dropout` | `Fusion/fused_bmm_rmsnorm_gelu_dropout.py` | bmm + RMSNorm + GELU + dropout | `fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.5, eps=1e-5, training=True, approximate='none')` |
| `expanded_pilot_norm_fused_bmm_rmsnorm_gelu_dropout_sub` | `fused_bmm_rmsnorm_gelu_dropout_sub` | `Fusion/fused_bmm_rmsnorm_gelu_dropout_sub.py` | bmm + RMSNorm + GELU + dropout + sub | `fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5)` |

## Matmul

| task_id | KBX op | KBX task file | dtype/profile | required entrypoint |
|---|---|---|---|---|
| `expanded_pilot_matmul_matmul` | `matmul` | `MatrixMultiply/matmul.py` | general torch.matmul cases | `matmul(tensor1, tensor2)` |
| `expanded_pilot_matmul_matmul_fp16` | `matmul_fp16` | `MatrixMultiply/matmul_fp16.py` | fp16 matmul, including batched cases | `matmul_fp16(input, other)` |
| `expanded_pilot_matmul_matmul_bf16` | `matmul_bf16` | `MatrixMultiply/matmul_bf16.py` | bf16 matmul, including batched cases | `matmul_bf16(input, other)` |
| `expanded_pilot_matmul_addmm` | `addmm` | `MatrixMultiply/addmm.py` | GEMM + scaled add | `addmm(input, mat1, mat2, beta=1, alpha=1, out=None)` |
| `expanded_pilot_matmul_matrix_vector_dot` | `matrix_vector_dot` | `MatrixMultiply/matrix_vector_dot.py` | matvec + scaled accumulation | `matrix_vector_dot(A, x, y, alpha, beta)` |
| `expanded_pilot_matmul_tril_mm_and_scale` | `tril_mm_and_scale` | `MatrixMultiply/tril_mm_and_scale.py` | triangular mask + matmul + scale | `tril_mm_and_scale(A, B, alpha, beta)` |

## Layout / Indexing / Irregular Memory

| task_id | KBX op | KBX task file | dtype/profile | required entrypoint |
|---|---|---|---|---|
| `expanded_pilot_layout_index_select` | `index_select` | `Index/index_select.py` | gather along selected dim | `index_select(input, dim, index)` |
| `expanded_pilot_layout_permute_copy` | `permute_copy` | `Index/permute_copy.py` | permute + clone/copy | `permute_copy(input, dims)` |
| `expanded_pilot_layout_scatter` | `scatter` | `Index/scatter.py` | scatter writes | `scatter(input, dim, index, src)` |
| `expanded_pilot_layout_masked_select` | `masked_select` | `Index/masked_select.py` | boolean mask compaction | `masked_select(input, mask)` |
| `expanded_pilot_layout_expand_where` | `expand_where` | `Index/expand_where.py` | broadcast expand + where | `expand_where(input, target_sizes, cond, other)` |
| `expanded_pilot_layout_fused_gather_masked_fill` | `fused_gather_masked_fill` | `Fusion/fused_gather_masked_fill.py` | gather + masked fill | `fused_gather_masked_fill(input, dim, index, mask, value)` |
