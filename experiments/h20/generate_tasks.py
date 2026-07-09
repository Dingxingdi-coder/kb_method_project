#!/usr/bin/env python3
"""Generate public task.json files and private hidden-test files for H20 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import safe_id, short_hash, write_json  # noqa: E402


ROUND1 = [
    ("reduction", "sum", [1024, 64], "fp32"),
    ("reduction", "sum", [1024, 257], "fp16"),
    ("reduction", "max", [4096, 1024], "fp16"),
    ("reduction", "sum", [8192, 4096], "bf16"),
    ("reduction", "max", [128, 8192], "fp32"),
    ("softmax", "row_softmax", [1024, 128], "fp16"),
    ("softmax", "row_softmax", [1024, 257], "fp16"),
    ("softmax", "row_softmax", [4096, 1024], "bf16"),
    ("softmax", "row_softmax", [8192, 2048], "fp16"),
    ("softmax", "row_softmax", [512, 4096], "fp32"),
    ("layernorm", "layernorm_forward", [1024, 768], "fp16"),
    ("layernorm", "layernorm_forward", [2048, 1024], "bf16"),
    ("layernorm", "layernorm_forward", [4096, 2048], "fp16"),
    ("layernorm", "layernorm_forward", [8192, 4096], "bf16"),
    ("layernorm", "layernorm_forward", [1024, 8192], "fp16"),
    ("matmul", "matmul", {"M": 128, "N": 128, "K": 128}, "fp16"),
    ("matmul", "matmul", {"M": 512, "N": 512, "K": 512}, "bf16"),
    ("matmul", "matmul", {"M": 1024, "N": 1024, "K": 1024}, "fp16"),
    ("matmul", "matmul", {"M": 4096, "N": 256, "K": 1024}, "fp16"),
    ("matmul", "matmul", {"M": 256, "N": 4096, "K": 1024}, "bf16"),
]

ROUND2 = [
    ("softmax", "row_softmax", [2048, 513], "fp16"),
    ("softmax", "row_softmax", [16384, 1024], "bf16"),
    ("layernorm", "layernorm_forward", [4096, 3072], "fp16"),
    ("layernorm", "layernorm_forward", [2048, 6144], "bf16"),
    ("reduction", "sum", [2048, 1536], "fp16"),
    ("reduction", "max", [16384, 255], "fp32"),
    ("matmul", "matmul", {"M": 768, "N": 768, "K": 768}, "fp16"),
    ("matmul", "matmul", {"M": 2048, "N": 512, "K": 1536}, "bf16"),
]

def kbx_task(
    category: str,
    op_name: str,
    task_file: str,
    signature: str,
    shape: Any,
    dtype: str,
    semantics: str,
) -> dict[str, Any]:
    category_prefix = "norm" if category == "layernorm" else category
    return {
        "task_id": f"expanded_pilot_{category_prefix}_{op_name}",
        "op_family": category,
        "op_name": op_name,
        "variant": op_name,
        "shape": shape,
        "dtype": dtype,
        "kernelbenchx": {
            "task_file": task_file,
            "entrypoint": op_name,
            "signature": signature,
        },
        "semantics": semantics,
    }


EXPANDED_PILOT = [
    kbx_task("pointwise", "add", "Math/add.py", "add(input, other, alpha=1, out=None)", [1024, 1024], "fp32", "Elementwise add with tensor/scalar other, broadcasting, alpha, and optional out semantics."),
    kbx_task("pointwise", "mul", "Math/mul.py", "mul(input, other, out=None)", [1024, 1024], "fp32", "Elementwise multiply with tensor/scalar other and broadcasting semantics."),
    kbx_task("pointwise", "gelu_fp16", "Activation/gelu_fp16.py", "gelu_fp16(input, approximate='none')", [1024, 1024], "fp16", "FP16 GELU activation with exact and tanh approximation modes."),
    kbx_task("pointwise", "gelu_bf16", "Activation/gelu_bf16.py", "gelu_bf16(input, approximate='none')", [2048, 512], "bf16", "BF16 GELU activation with exact and tanh approximation modes."),
    kbx_task("pointwise", "fused_add_gelu", "Fusion/fused_add_gelu.py", "fused_add_gelu(input, other, alpha=1, approximate='none', out=None)", [512, 512], "fp32", "Fused add followed by GELU, including scalar/tensor other and approximation mode."),
    kbx_task("pointwise", "fused_mul_sub", "Fusion/fused_mul_sub.py", "fused_mul_sub(input, other_mul, other_sub, alpha=1, out=None)", [512, 512], "fp32", "Fused elementwise multiply and scaled subtract with tensor/scalar operands."),
    kbx_task("reduction", "sum", "Reduce/sum.py", "sum(input, dim, keepdim=False, dtype=None)", [128, 256], "fp32", "Sum over scalar or tuple dims with keepdim and optional dtype."),
    kbx_task("reduction", "mean", "Reduce/mean.py", "mean(input_tensor, dim, keepdim=False, dtype=None, out=None)", [128, 256], "fp32", "Mean over scalar or tuple dims with keepdim and optional dtype."),
    kbx_task("reduction", "std", "Reduce/std.py", "std(input, dim=None, correction=1, keepdim=False, out=None)", [128, 256], "fp32", "Standard deviation over optional dims with correction and keepdim."),
    kbx_task("reduction", "min", "Reduce/min.py", "min(input_tensor, dim, keepdim=False)", [128, 256], "fp32", "Value and index minimum reduction along a requested dimension."),
    kbx_task("reduction", "argmax", "Reduce/argmax.py", "argmax(input_tensor, dim, keepdim=False)", [128, 256], "fp32", "Argmax index reduction along a requested dimension."),
    kbx_task("reduction", "fused_sum_std", "Fusion/fused_sum_std.py", "fused_sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None)", [128, 256], "fp32", "Sum along optional dims followed by standard deviation of the summed values."),
    kbx_task("softmax", "softmax", "Math/softmax.py", "softmax(input, dim, dtype=None)", [128, 256], "fp32", "Softmax along dim with optional dtype."),
    kbx_task("softmax", "fused_softmax_log", "Fusion/fused_softmax_log.py", "fused_softmax_log(input, dim=-1, dtype=None)", [128, 256], "fp32", "Log transform of positive inputs followed by softmax."),
    kbx_task("softmax", "fused_log_softmax_linear", "Fusion/fused_log_softmax_linear.py", "fused_log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None)", {"B": 64, "IN": 128, "OUT": 256}, "fp16", "Linear projection followed by log_softmax."),
    kbx_task("softmax", "fused_repeat_interleave_log_softmax", "Fusion/fused_repeat_interleave_log_softmax.py", "fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None)", [64, 128], "fp32", "repeat_interleave followed by log_softmax."),
    kbx_task("softmax", "fused_cross_entropy_log_softmax", "Fusion/fused_cross_entropy_log_softmax.py", "fused_cross_entropy_log_softmax(input, target, dim=1, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)", {"B": 64, "C": 100}, "fp32", "Cross entropy over logits after log_softmax transform."),
    kbx_task("softmax", "attention", "Fusion/attention.py", "attention(q, k, v, causal=False, softmax_scale=None, *, out=None)", {"B": 2, "H": 4, "S": 64, "D": 32}, "fp16", "Scaled dot-product attention with optional causal mask and explicit scale."),
    kbx_task("layernorm", "layernorm_w8a8", "Quantization/layernorm_w8a8.py", "layernorm_w8a8(input, normalized_shape, weight=None, bias=None, eps=1e-5)", {"B": 32, "D": 256}, "fp32", "LayerNorm-like dynamic W8A8 pipeline semantics with fp32 input/output."),
    kbx_task("layernorm", "fused_layer_norm_relu_linear", "Fusion/fused_layer_norm_relu_linear.py", "fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True)", {"B": 64, "IN": 128, "OUT": 256}, "fp32", "Linear projection, ReLU, then layer_norm over the output feature dimension."),
    kbx_task("layernorm", "fused_cross_entropy_softmax_layernorm", "Fusion/fused_cross_entropy_softmax_layernorm.py", "fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None)", {"B": 32, "C": 128}, "fp32", "Cross entropy plus softmax and layer_norm output tuple."),
    kbx_task("layernorm", "fused_silu_layer_norm_conv2d", "Fusion/fused_silu_layer_norm_conv2d.py", "fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5)", {"B": 2, "C": 3, "H": 32, "W": 32, "OC": 8, "K": 3}, "fp32", "conv2d, layer_norm over output channels/spatial dimensions, then SiLU."),
    kbx_task("layernorm", "fused_bmm_rmsnorm_gelu_dropout", "Fusion/fused_bmm_rmsnorm_gelu_dropout.py", "fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.5, eps=1e-5, training=True, approximate='none')", {"B": 4, "M": 16, "K": 32, "N": 64}, "fp32", "BMM, RMSNorm, GELU, and dropout semantics."),
    kbx_task("layernorm", "fused_bmm_rmsnorm_gelu_dropout_sub", "Fusion/fused_bmm_rmsnorm_gelu_dropout_sub.py", "fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5)", {"B": 4, "M": 16, "K": 32, "N": 64}, "fp32", "BMM, RMSNorm, GELU, dropout, and subtraction-task entrypoint semantics."),
    kbx_task("matmul", "matmul", "MatrixMultiply/matmul.py", "matmul(tensor1, tensor2)", {"M": 64, "N": 32, "K": 128}, "fp32", "General torch.matmul cases including matrix and batched inputs."),
    kbx_task("matmul", "matmul_fp16", "MatrixMultiply/matmul_fp16.py", "matmul_fp16(input, other)", {"M": 64, "N": 64, "K": 128}, "fp16", "FP16 matmul with matrix and batched inputs."),
    kbx_task("matmul", "matmul_bf16", "MatrixMultiply/matmul_bf16.py", "matmul_bf16(input, other)", {"M": 64, "N": 64, "K": 128}, "bf16", "BF16 matmul with matrix and batched inputs."),
    kbx_task("matmul", "addmm", "MatrixMultiply/addmm.py", "addmm(input, mat1, mat2, beta=1, alpha=1, out=None)", {"M": 64, "N": 32, "K": 128}, "fp32", "Matrix multiplication plus scaled input add."),
    kbx_task("matmul", "matrix_vector_dot", "MatrixMultiply/matrix_vector_dot.py", "matrix_vector_dot(A, x, y, alpha, beta)", {"N": 128}, "fp32", "Matrix-vector update followed by dot product."),
    kbx_task("matmul", "tril_mm_and_scale", "MatrixMultiply/tril_mm_and_scale.py", "tril_mm_and_scale(A, B, alpha, beta)", {"N": 128, "P": 32}, "fp32", "Lower-triangular matrix multiply followed by scaling."),
    kbx_task("layout", "index_select", "Index/index_select.py", "index_select(input, dim, index)", {"B": 8, "N": 32, "D": 256, "K": 64}, "fp32", "Gather selected indices along a requested dimension."),
    kbx_task("layout", "permute_copy", "Index/permute_copy.py", "permute_copy(input, dims)", [4, 8, 16, 32], "fp32", "Permute dimensions and materialize a fresh copied tensor."),
    kbx_task("layout", "scatter", "Index/scatter.py", "scatter(input, dim, index, src)", {"B": 4, "N": 128, "D": 256, "K": 64}, "fp32", "Scatter source values into an input tensor along a requested dimension."),
    kbx_task("layout", "masked_select", "Index/masked_select.py", "masked_select(input, mask)", [64, 128], "fp32", "Select elements under a boolean mask, returning a compact 1D tensor."),
    kbx_task("layout", "expand_where", "Index/expand_where.py", "expand_where(input, target_sizes, cond, other)", {"B": 64, "N": 512}, "fp32", "Broadcast expand a singleton input and select with torch.where semantics."),
    kbx_task("layout", "fused_gather_masked_fill", "Fusion/fused_gather_masked_fill.py", "fused_gather_masked_fill(input, dim, index, mask, value)", {"B": 64, "N": 128, "K": 32}, "fp32", "Gather along a dimension followed by masked fill."),
]


def shape_id(shape: Any) -> str:
    if isinstance(shape, dict):
        preferred = ["M", "N", "K", "B", "rows", "width", "out_rows", "vocab", "dim", "batch", "seq", "nnz", "cut"]
        parts = [f"{key.lower()}{shape[key]}" for key in preferred if key in shape]
        parts.extend(f"{key.lower()}{value}" for key, value in sorted(shape.items()) if key not in preferred)
        return "_".join(parts)
    return "x".join(str(x) for x in shape)


def hidden_variants(op_family: str, shape: Any, dtype: str) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    if op_family in {"pointwise", "reduction", "softmax", "layernorm"}:
        b, n = shape
        for idx, candidate_shape in enumerate([[max(1, b // 2 + 1), n + 1 if n < 8192 else n - 1], [b + 7, max(1, n - 3)]]):
            tests.append({"shape": candidate_shape, "dtype": dtype, "layout": "contiguous", "seed_offset": idx + 100})
        tests.append({"shape": shape, "dtype": dtype, "layout": "non_contiguous", "seed_offset": 200})
    elif op_family == "matmul":
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        tests.extend([
            {"shape": {"M": max(16, m // 2 + 17), "N": n, "K": k}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
            {"shape": {"M": m, "N": max(16, n // 2 + 19), "K": max(16, k // 2 + 23)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 101},
        ])
    elif op_family == "layout":
        if "M" in shape and "N" in shape:
            tests.extend([
                {"shape": {"M": max(16, int(shape["M"]) // 2 + 17), "N": int(shape["N"]) + 1}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": {"M": int(shape["M"]) + 7, "N": max(16, int(shape["N"]) - 3)}, "dtype": dtype, "layout": "non_contiguous", "seed_offset": 101},
            ])
        elif "rows" in shape:
            tests.extend([
                {"shape": {"rows": max(64, int(shape["rows"]) // 2 + 17), "width": int(shape["width"]) + 1, "out_rows": max(32, int(shape["out_rows"]) // 2 + 9)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "index_int32", "seed_offset": 101},
            ])
        elif "vocab" in shape:
            tests.extend([
                {"shape": {"vocab": max(128, int(shape["vocab"]) // 2 + 19), "dim": int(shape["dim"]) + 1, "batch": max(16, int(shape["batch"]) // 2 + 5), "seq": int(shape["seq"]) + 1}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "index_int32", "seed_offset": 101},
            ])
        elif "nnz" in shape:
            tests.extend([
                {"shape": {"nnz": max(64, int(shape["nnz"]) // 2 + 13), "width": int(shape["width"]) + 1, "out_rows": max(16, int(shape["out_rows"]) // 2 + 7)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "collision_heavy", "seed_offset": 101},
            ])
        elif "B" in shape:
            tests.extend([
                {"shape": {"B": max(16, int(shape["B"]) // 2 + 11), "N": int(shape["N"]) + 1, "cut": max(1, int(shape["cut"]) // 2)}, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
                {"shape": shape, "dtype": dtype, "layout": "non_contiguous", "seed_offset": 101},
            ])
    return tests


def op_spec(op_family: str, variant: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if metadata and metadata.get("kernelbenchx"):
        kbx = metadata["kernelbenchx"]
        entrypoint = kbx["entrypoint"]
        return {
            "signature": metadata.get("semantics", variant),
            "candidate_interface": f"candidate(*args, **kwargs) -> same result as `{entrypoint}(...)`; `{kbx['signature']}` must also exist",
            "semantics": metadata.get("semantics", "Implement the declared KernelBench-X operator semantics."),
            "tolerance": "dtype-dependent",
            "kernelbenchx_entrypoint": entrypoint,
            "kernelbenchx_signature": kbx["signature"],
            "kernelbenchx_task_file": kbx["task_file"],
            "target_banned_api": ["Do not call the target torch/PyTorch convenience API as the main submitted computation."],
        }
    if op_family == "pointwise":
        interfaces = {
            "bias_gelu": "candidate(x, bias) -> y",
            "bias_silu": "candidate(x, bias) -> y",
            "residual_relu": "candidate(x, residual) -> y",
            "broadcast_affine": "candidate(x, scale, bias) -> y",
            "gated_silu_multiply": "candidate(x1, x2) -> y",
            "clamp_mul_add": "candidate(x, scale, bias, min_value, max_value) -> y",
        }
        signatures = {
            "bias_gelu": "y = gelu(x + bias)",
            "bias_silu": "y = silu(x + bias)",
            "residual_relu": "y = relu(x + residual)",
            "broadcast_affine": "y = x * scale + bias",
            "gated_silu_multiply": "y = silu(x1) * x2",
            "clamp_mul_add": "y = clamp(x, min_value, max_value) * scale + bias",
        }
        return {"signature": signatures.get(variant, variant), "candidate_interface": interfaces.get(variant, "candidate(*args) -> y"), "semantics": "Implement the fused elementwise expression; support broadcast vectors over the last dimension and preserve output dtype.", "tolerance": "dtype-dependent", "target_banned_api": ["torch.nn.functional.gelu", "torch.nn.functional.silu", "torch.gelu", "torch.silu", "torch.clamp as the main submitted computation"]}
    if op_family == "softmax":
        return {"signature": "y = torch.softmax(x, dim=-1)", "candidate_interface": "candidate(x) -> y", "semantics": "Subtract row max before exp; mask padded tail elements.", "tolerance": "dtype-dependent"}
    if op_family == "reduction":
        return {"signature": f"y = torch.{variant}(x, dim=-1)", "candidate_interface": "candidate(x, reduce_op) -> y", "semantics": "Reduce the last dimension with fp32 accumulation for sum unless declared otherwise.", "tolerance": "dtype-dependent"}
    if op_family == "layernorm":
        return {"signature": "y = (x - mean) / sqrt(var + eps) * gamma + beta", "candidate_interface": "candidate(x, gamma, beta, eps) -> y", "semantics": "Mean and variance over last dimension; fp32 accumulation.", "tolerance": "dtype-dependent"}
    if op_family == "matmul":
        return {"signature": "C = A @ B", "candidate_interface": "candidate(a, b) -> c", "semantics": "Matrix multiplication with declared dtype and backend default/fp32 accumulation.", "tolerance": "dtype-dependent"}
    if op_family == "layout":
        interfaces = {
            "transpose_copy": "candidate(x) -> y",
            "gather_rows": "candidate(x, indices) -> y",
            "embedding_lookup": "candidate(weight, indices) -> y",
            "scatter_add": "candidate(src, indices, out_rows) -> y",
            "slice_concat": "candidate(a, b, cut) -> y",
            "strided_copy": "candidate(x) -> y",
        }
        signatures = {
            "transpose_copy": "y = x.T.contiguous()",
            "gather_rows": "y = x[indices]",
            "embedding_lookup": "y = weight[indices]",
            "scatter_add": "y = zeros(out_rows, width).scatter_add(dim=0, index=indices, src=src)",
            "slice_concat": "y = torch.cat([a[:, :cut], b[:, cut:]], dim=-1)",
            "strided_copy": "y = x.contiguous() for a strided input view",
        }
        return {"signature": signatures.get(variant, variant), "candidate_interface": interfaces.get(variant, "candidate(*args) -> y"), "semantics": "Implement the declared layout or irregular-memory transformation without reading hidden cases; handle bounds, strides, index dtype, and collisions where applicable.", "tolerance": "dtype-dependent", "target_banned_api": ["torch.transpose/permute/contiguous as the submitted target copy", "torch.gather/index_select/embedding as the submitted target gather", "torch.scatter_add as the submitted target scatter", "torch.cat as the submitted target concat"]}
    return {}


def kbx_hidden_variants(shape: Any, dtype: str) -> list[dict[str, Any]]:
    if isinstance(shape, list):
        if len(shape) == 1:
            alt_shape: Any = [max(1, int(shape[0]) + 17)]
        else:
            alt_shape = [max(1, int(shape[0]) // 2 + 7), max(1, int(shape[1]) + 1), *shape[2:]]
    elif isinstance(shape, dict):
        alt_shape = {key: max(1, int(value) + (1 if idx % 2 else 7)) for idx, (key, value) in enumerate(shape.items())}
    else:
        alt_shape = shape
    return [
        {"shape": alt_shape, "dtype": dtype, "layout": "contiguous", "seed_offset": 100},
        {"shape": shape, "dtype": dtype, "layout": "alternate_args", "seed_offset": 200},
    ]


def make_task(
    round_name: str,
    op_family: str,
    variant: str,
    shape: Any,
    dtype: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = metadata.get("task_id") if metadata else None
    task_id = str(task_id or safe_id(f"{round_name}_{op_family}_{variant}_{shape_id(shape)}_{dtype}"))
    task = {
        "schema_version": "0.1",
        "task_id": task_id,
        "round": round_name,
        "op_family": op_family,
        "op_name": variant,
        "variant": variant,
        "shape": shape,
        "dtype": dtype,
        "op_spec": op_spec(op_family, variant, metadata),
        "op_spec_hash": short_hash([op_family, variant, shape, dtype]),
        "goal": "compile_correct_and_benchmark",
        "public_tests": [{"shape": shape, "dtype": dtype, "layout": "contiguous", "seed_offset": 0}],
        "budget": {"max_iterations": 8, "max_wall_time_minutes": 60, "warmup": 100, "repeats": 500},
        "agent_allowed_files": ["candidate.py", "notes.md"],
        "agent_forbidden_files": ["reference.py", "hidden_tests.json", "experiments/h20/harness.py"],
    }
    if metadata and metadata.get("kernelbenchx"):
        task["kernelbenchx"] = metadata["kernelbenchx"]
        hidden_tests = kbx_hidden_variants(shape, dtype)
    else:
        hidden_tests = hidden_variants(op_family, shape, dtype)
    hidden = {"schema_version": "0.1", "task_id": task_id, "hidden_tests": hidden_tests, "notes": "Do not expose this file to the Coding Agent."}
    return task, hidden


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=["round1", "round2", "expanded_pilot"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--operators", default=None, help="Accepted for protocol traceability; this script uses the encoded task table.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    hidden_dir = out_dir / "_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    if args.round == "round1":
        rows = ROUND1
    elif args.round == "round2":
        rows = ROUND2
    else:
        rows = EXPANDED_PILOT
    manifest: list[dict[str, str]] = []

    for row in rows:
        if isinstance(row, dict):
            task, hidden = make_task(args.round, row["op_family"], row["op_name"], row["shape"], row["dtype"], row)
        else:
            op_family, variant, shape, dtype = row
            task, hidden = make_task(args.round, op_family, variant, shape, dtype)
        task_path = out_dir / f"{task['task_id']}.json"
        hidden_path = hidden_dir / f"{task['task_id']}.hidden.json"
        write_json(task_path, task)
        write_json(hidden_path, hidden)
        manifest.append({"task_id": task["task_id"], "task": str(task_path), "hidden": str(hidden_path)})

    write_json(out_dir / "manifest.json", {"round": args.round, "tasks": manifest})
    print(f"wrote {len(manifest)} tasks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
