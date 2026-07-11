#!/usr/bin/env python3
"""Fixed H20 MVP harness for Triton/PyTorch candidate kernels.

Candidate interface:
- pointwise: variant-specific, see task.json op_spec.candidate_interface
- reduction: candidate(x, reduce_op) -> y
- softmax: candidate(x) -> y
- layernorm: candidate(x, gamma, beta, eps) -> y
- matmul: candidate(a, b) -> c
- layout: variant-specific, see task.json op_spec.candidate_interface
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ecc_utils import read_json, sha256_file, short_hash, utc_now, write_json  # noqa: E402


def torch_dtype(name: str):
    import torch
    return {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[name]


def tolerance(dtype: str, op_family: str) -> tuple[float, float]:
    if dtype in ("fp32", "float32"):
        return 1e-4, 1e-4
    if dtype in ("bf16", "bfloat16"):
        return 8e-2, 8e-2
    if op_family in ("reduction", "layernorm"):
        return 5e-2, 5e-2
    return 3e-2, 3e-2


def load_candidate(path: Path, entrypoint: str | None = None):
    spec = importlib.util.spec_from_file_location("ecc_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ecc_candidate"] = module
    spec.loader.exec_module(module)
    if entrypoint:
        fn = getattr(module, entrypoint, None)
        if callable(fn):
            return fn
        raise AttributeError(f"candidate.py must define callable {entrypoint}() for this KernelBench-X-aligned task")
    for name in ("candidate", "run", "forward"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError("candidate.py must define callable candidate(), run(), or forward()")


def make_tensor_2d(shape: list[int], dtype_name: str, seed: int, layout: str, device: str):
    import torch
    torch.manual_seed(seed)
    b, n = int(shape[0]), int(shape[1])
    dtype = torch_dtype(dtype_name)
    if layout == "non_contiguous":
        return torch.randn((n, b), device=device, dtype=dtype).t()
    return torch.randn((b, n), device=device, dtype=dtype)


def make_vector(length: int, dtype_name: str, seed: int, device: str, scale: float = 1.0):
    import torch
    torch.manual_seed(seed)
    return torch.randn((int(length),), device=device, dtype=torch_dtype(dtype_name)) * scale


def make_strided_2d(m: int, n: int, dtype_name: str, seed: int, device: str, row_stride: int = 2):
    import torch
    torch.manual_seed(seed)
    base = torch.randn((int(m) * int(row_stride), int(n)), device=device, dtype=torch_dtype(dtype_name))
    return base[:: int(row_stride)]


def is_kbx_task(task: dict[str, Any]) -> bool:
    return isinstance(task.get("kernelbenchx"), dict) and bool(task["kernelbenchx"].get("entrypoint"))


def clone_input_tree(value: Any) -> Any:
    import torch
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_input_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_input_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_input_tree(item) for key, item in value.items()}
    return value


def rand(shape: Any, dtype_name: str, seed: int, device: str, positive: bool = False):
    import torch
    torch.manual_seed(seed)
    tensor = torch.randn(tuple(int(x) for x in shape), device=device, dtype=torch_dtype(dtype_name))
    return tensor.abs() + 0.1 if positive else tensor


def kbx_make_inputs(task: dict[str, Any], test: dict[str, Any], base_seed: int, device: str):
    import torch
    op = str(task.get("op_name"))
    dtype_name = test.get("dtype", task.get("dtype", "fp32"))
    layout = test.get("layout", "contiguous")
    seed = base_seed + int(test.get("seed_offset", 0))
    shape = test.get("shape", task["shape"])
    alternate = layout == "alternate_args"

    if op in {"add", "mul", "gelu_fp16", "gelu_bf16", "fused_add_gelu", "fused_mul_sub"}:
        x = rand(shape, dtype_name, seed, device)
        if op == "add":
            other = 0.5 if alternate else rand(shape, dtype_name, seed + 1, device)
            return x, other, (0.5 if alternate else 1)
        if op == "mul":
            other = 2.5 if alternate else rand(shape, dtype_name, seed + 1, device)
            return x, other
        if op in {"gelu_fp16", "gelu_bf16"}:
            return x, ("tanh" if alternate else "none")
        if op == "fused_add_gelu":
            other = 0.5 if alternate else rand(shape, dtype_name, seed + 1, device)
            return x, other, (0.5 if alternate else 1), ("tanh" if alternate else "none")
        y = 2.0 if alternate else rand(shape, dtype_name, seed + 1, device)
        z = 0.5 if alternate else rand(shape, dtype_name, seed + 2, device)
        return x, y, z, (-0.5 if alternate else 1)

    if op in {"sum", "mean", "std", "min", "argmax", "fused_sum_std", "softmax", "fused_softmax_log"}:
        positive = op == "fused_softmax_log"
        x = rand(shape, dtype_name, seed, device, positive=positive)
        dim = 0 if alternate else -1
        keepdim = bool(alternate and op in {"sum", "mean", "std", "min", "argmax", "fused_sum_std"})
        if op == "sum":
            return x, dim, keepdim, None
        if op == "mean":
            return x, dim, keepdim, None
        if op == "std":
            return x, dim, (0 if alternate else 1), keepdim
        if op == "min":
            return x, dim, keepdim
        if op == "argmax":
            return x, dim, keepdim
        if op == "fused_sum_std":
            return x, dim, keepdim, None, (0 if alternate else 1)
        if op == "softmax":
            return x, dim, None
        return x, dim, None

    if op == "fused_log_softmax_linear":
        b, in_features, out_features = int(shape["B"]), int(shape["IN"]), int(shape["OUT"])
        x = rand((b, in_features), dtype_name, seed, device)
        weight = rand((out_features, in_features), dtype_name, seed + 1, device)
        bias = None if alternate else rand((out_features,), dtype_name, seed + 2, device)
        return x, weight, bias, -1, None
    if op == "fused_repeat_interleave_log_softmax":
        x = rand(shape, dtype_name, seed, device)
        if alternate:
            repeats = torch.randint(1, 4, (int(shape[0]),), device=device, dtype=torch.int64)
            return x[:, 0], repeats, 0
        return x, 2, 1
    if op == "fused_cross_entropy_log_softmax":
        b, c = int(shape["B"]), int(shape["C"])
        logits = rand((b, c), dtype_name, seed, device)
        torch.manual_seed(seed + 1)
        target = torch.randint(0, c, (b,), device=device, dtype=torch.int64)
        weight = rand((c,), dtype_name, seed + 2, device).abs() if alternate else None
        return logits, target, 1, weight, -100, ("sum" if alternate else "mean"), (0.1 if alternate else 0.0)
    if op == "attention":
        b, h, s, d = int(shape["B"]), int(shape["H"]), int(shape["S"]), int(shape["D"])
        q = rand((b, h, s, d), dtype_name, seed, device)
        k = rand((b, h, s, d), dtype_name, seed + 1, device)
        v = rand((b, h, s, d), dtype_name, seed + 2, device)
        return q, k, v, bool(alternate), (0.125 if alternate else None)

    if op == "layernorm_w8a8":
        b, d = int(shape["B"]), int(shape["D"])
        x = rand((b, d), "fp32", seed, device).clamp(-10, 10)
        weight = None if alternate else rand((d,), "fp32", seed + 1, device)
        bias = None if alternate else rand((d,), "fp32", seed + 2, device)
        return x, (d,), weight, bias, (1e-3 if alternate else 1e-5)
    if op == "fused_layer_norm_relu_linear":
        b, in_features, out_features = int(shape["B"]), int(shape["IN"]), int(shape["OUT"])
        x = rand((b, in_features), dtype_name, seed, device)
        weight = rand((out_features, in_features), dtype_name, seed + 1, device)
        bias = None if alternate else rand((out_features,), dtype_name, seed + 2, device)
        return x, weight, bias, out_features, (1e-3 if alternate else 1e-5), True
    if op == "fused_cross_entropy_softmax_layernorm":
        b, c = int(shape["B"]), int(shape["C"])
        logits = rand((b, c), dtype_name, seed, device)
        torch.manual_seed(seed + 1)
        targets = torch.randint(0, c, (b,), device=device, dtype=torch.int64)
        weight = rand((c,), dtype_name, seed + 2, device).abs() if alternate else None
        return logits, targets, c, weight, -100, "mean", (0.1 if alternate else 0.0), (1e-3 if alternate else 1e-5)
    if op == "fused_silu_layer_norm_conv2d":
        b, c, h, w, oc, ksize = int(shape["B"]), int(shape["C"]), int(shape["H"]), int(shape["W"]), int(shape["OC"]), int(shape["K"])
        x = rand((b, c, h, w), dtype_name, seed, device)
        conv_weight = rand((oc, c, ksize, ksize), dtype_name, seed + 1, device)
        conv_bias = None if alternate else rand((oc,), dtype_name, seed + 2, device)
        return x, None, conv_weight, conv_bias, 1, 1, 1, 1, (1e-3 if alternate else 1e-5)
    if op in {"fused_bmm_rmsnorm_gelu_dropout", "fused_bmm_rmsnorm_gelu_dropout_sub"}:
        b, m, kdim, n = int(shape["B"]), int(shape["M"]), int(shape["K"]), int(shape["N"])
        input1 = rand((b, m, kdim), dtype_name, seed, device)
        input2 = rand((b, kdim, n), dtype_name, seed + 1, device)
        if op == "fused_bmm_rmsnorm_gelu_dropout":
            return input1, input2, n, 0.1, (1e-3 if alternate else 1e-5), False, ("tanh" if alternate else "none")
        other = rand((b, m, n), dtype_name, seed + 2, device)
        return input1, input2, other, n, 0.1, False, ("tanh" if alternate else "none"), (1e-3 if alternate else 1e-5)

    if op in {"matmul", "matmul_fp16", "matmul_bf16"}:
        dtype = dtype_name
        if alternate:
            b, m, n, kdim = 4, int(shape["M"]), int(shape["N"]), int(shape["K"])
            return rand((b, m, kdim), dtype, seed, device), rand((b, kdim, n), dtype, seed + 1, device)
        return rand((int(shape["M"]), int(shape["K"])), dtype, seed, device), rand((int(shape["K"]), int(shape["N"])), dtype, seed + 1, device)
    if op == "addmm":
        m, n, kdim = int(shape["M"]), int(shape["N"]), int(shape["K"])
        return rand((m, n), dtype_name, seed, device), rand((m, kdim), dtype_name, seed + 1, device), rand((kdim, n), dtype_name, seed + 2, device), (0.5 if alternate else 1), (2.0 if alternate else 1)
    if op == "matrix_vector_dot":
        n = int(shape["N"])
        return rand((n, n), dtype_name, seed, device), rand((n,), dtype_name, seed + 1, device), rand((n,), dtype_name, seed + 2, device), (0.5 if alternate else 1.0), (0.5 if alternate else 0.0)
    if op == "tril_mm_and_scale":
        n, p = int(shape["N"]), int(shape["P"])
        return rand((n, n), dtype_name, seed, device), rand((n, p), dtype_name, seed + 1, device), 1.0, (0.5 if alternate else 1.0)

    if op == "index_select":
        b, n, d, k = int(shape["B"]), int(shape["N"]), int(shape["D"]), int(shape["K"])
        x = rand((b, n, d), dtype_name, seed, device)
        dim = 1 if alternate else 2
        high = n if dim == 1 else d
        torch.manual_seed(seed + 1)
        index = torch.randint(0, high, (k,), device=device, dtype=torch.int64)
        return x, dim, index
    if op == "permute_copy":
        x = rand(shape, dtype_name, seed, device)
        dims = [3, 0, 2, 1] if len(shape) == 4 and alternate else list(reversed(range(len(shape))))
        return x, dims
    if op == "scatter":
        b, n, d, k = int(shape["B"]), int(shape["N"]), int(shape["D"]), int(shape["K"])
        base = rand((b, n, d), dtype_name, seed, device)
        torch.manual_seed(seed + 1)
        index = torch.randint(0, d, (b, n, k), device=device, dtype=torch.int64)
        src = rand((b, n, k), dtype_name, seed + 2, device)
        return base, 2, index, src
    if op == "masked_select":
        x = rand(shape, dtype_name, seed, device)
        torch.manual_seed(seed + 1)
        mask_shape = (1, int(shape[1])) if alternate and len(shape) == 2 else tuple(shape)
        mask = torch.rand(mask_shape, device=device) > 0.5
        return x, mask
    if op == "expand_where":
        b, n = int(shape["B"]), int(shape["N"])
        x = rand((1, n), dtype_name, seed, device)
        torch.manual_seed(seed + 1)
        cond = torch.rand((b, 1), device=device) > 0.5
        other = rand((b, n), dtype_name, seed + 2, device)
        return x, (b, n), cond, other
    if op == "fused_gather_masked_fill":
        b, n, k = int(shape["B"]), int(shape["N"]), int(shape["K"])
        x = rand((b, n), dtype_name, seed, device)
        torch.manual_seed(seed + 1)
        index = torch.randint(0, n, (b, k), device=device, dtype=torch.int64)
        torch.manual_seed(seed + 2)
        mask = torch.rand((b, k), device=device) > 0.5
        return x, 1, index, mask, -1.0

    raise ValueError(f"unsupported KernelBench-X op: {op}")


def kbx_reference(task: dict[str, Any], inputs: tuple[Any, ...]):
    import torch
    import torch.nn.functional as F
    op = str(task.get("op_name"))
    if op == "add":
        input_tensor, other, alpha = inputs
        return torch.add(input_tensor, other, alpha=alpha)
    if op == "mul":
        input_tensor, other = inputs
        return torch.mul(input_tensor, other)
    if op in {"gelu_fp16", "gelu_bf16"}:
        input_tensor, approximate = inputs
        return F.gelu(input_tensor, approximate=approximate)
    if op == "fused_add_gelu":
        input_tensor, other, alpha, approximate = inputs
        return F.gelu(input_tensor + alpha * other, approximate=approximate)
    if op == "fused_mul_sub":
        input_tensor, other_mul, other_sub, alpha = inputs
        return input_tensor * other_mul - alpha * other_sub
    if op == "sum":
        input_tensor, dim, keepdim, dtype = inputs
        return torch.sum(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype)
    if op == "mean":
        input_tensor, dim, keepdim, dtype = inputs
        return torch.mean(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype)
    if op == "std":
        input_tensor, dim, correction, keepdim = inputs
        return torch.std(input_tensor, dim=dim, correction=correction, keepdim=keepdim)
    if op == "min":
        input_tensor, dim, keepdim = inputs
        return torch.min(input_tensor, dim=dim, keepdim=keepdim)
    if op == "argmax":
        input_tensor, dim, keepdim = inputs
        return torch.argmax(input_tensor, dim=dim, keepdim=keepdim)
    if op == "fused_sum_std":
        input_tensor, dim, keepdim, dtype, correction = inputs
        summed = input_tensor.sum(dim=dim, keepdim=keepdim, dtype=dtype)
        n = summed.numel()
        mean = summed.mean()
        var = ((summed - mean) ** 2).sum()
        return (var / (n - correction)).sqrt() if n > correction else torch.tensor(0.0, dtype=summed.dtype, device=summed.device)
    if op == "softmax":
        input_tensor, dim, dtype = inputs
        return F.softmax(input_tensor, dim=dim, dtype=dtype)
    if op == "fused_softmax_log":
        input_tensor, dim, dtype = inputs
        if dtype is not None:
            input_tensor = input_tensor.to(dtype)
        return F.softmax(input_tensor.log(), dim=dim)
    if op == "fused_log_softmax_linear":
        input_tensor, weight, bias, dim, dtype = inputs
        output = torch.matmul(input_tensor, weight.T)
        if bias is not None:
            output = output + bias
        return F.log_softmax(output, dim=dim, dtype=dtype)
    if op == "fused_repeat_interleave_log_softmax":
        input_tensor, repeats, dim = inputs
        repeated = torch.repeat_interleave(input_tensor, repeats, dim=dim)
        return F.log_softmax(repeated, dim=dim)
    if op == "fused_cross_entropy_log_softmax":
        input_tensor, target, dim, weight, ignore_index, reduction, label_smoothing = inputs
        log_probs = F.log_softmax(input_tensor, dim=dim)
        return F.cross_entropy(log_probs, target, weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    if op == "attention":
        q, k, v, causal, softmax_scale = inputs
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * softmax_scale
        if causal:
            s = q.shape[2]
            mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        return torch.matmul(F.softmax(scores, dim=-1), v.float()).to(q.dtype)
    if op == "layernorm_w8a8":
        input_tensor, normalized_shape, weight, bias, eps = inputs
        return F.layer_norm(input_tensor, normalized_shape, weight, bias, eps)
    if op == "fused_layer_norm_relu_linear":
        input_tensor, weight, bias, normalized_shape, eps, _elementwise_affine = inputs
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        return F.layer_norm(F.relu(F.linear(input_tensor, weight, bias)), normalized_shape, eps=eps)
    if op == "fused_cross_entropy_softmax_layernorm":
        logits, targets, normalized_shape, weight, ignore_index, reduction, label_smoothing, eps = inputs
        loss = F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
        output = F.layer_norm(F.softmax(logits, dim=-1), normalized_shape=(normalized_shape,), eps=eps)
        return loss, output
    if op == "fused_silu_layer_norm_conv2d":
        x, _weight, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, ln_eps = inputs
        conv_out = F.conv2d(x, conv_weight, bias=conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
        return F.silu(F.layer_norm(conv_out, conv_out.shape[1:], eps=ln_eps))
    if op == "fused_bmm_rmsnorm_gelu_dropout":
        input1, input2, normalized_shape, dropout_p, eps, training, approximate = inputs
        z1 = torch.bmm(input1, input2)
        return F.dropout(F.gelu(F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps), approximate=approximate), p=dropout_p, training=training)
    if op == "fused_bmm_rmsnorm_gelu_dropout_sub":
        input1, input2, _other, normalized_shape, dropout_p, training, approximate, eps = inputs
        z1 = torch.bmm(input1, input2)
        return F.dropout(F.gelu(F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps), approximate=approximate), p=dropout_p, training=training)
    if op in {"matmul", "matmul_fp16", "matmul_bf16"}:
        input_tensor, other = inputs
        return torch.matmul(input_tensor, other)
    if op == "addmm":
        input_tensor, mat1, mat2, beta, alpha = inputs
        return torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)
    if op == "matrix_vector_dot":
        a, x, y, alpha, beta = inputs
        y_new = alpha * torch.mv(a, x) + beta * y
        y.copy_(y_new)
        return torch.dot(y, x)
    if op == "tril_mm_and_scale":
        a, b, alpha, beta = inputs
        return beta * (alpha * torch.mm(torch.tril(a), b))
    if op == "index_select":
        input_tensor, dim, index = inputs
        return torch.index_select(input_tensor, dim, index)
    if op == "permute_copy":
        input_tensor, dims = inputs
        return input_tensor.permute(dims).clone()
    if op == "scatter":
        input_tensor, dim, index, src = inputs
        return input_tensor.scatter(dim, index, src)
    if op == "masked_select":
        input_tensor, mask = inputs
        return torch.masked_select(input_tensor, mask)
    if op == "expand_where":
        input_tensor, target_sizes, cond, other = inputs
        return torch.where(cond, input_tensor.expand(*target_sizes), other)
    if op == "fused_gather_masked_fill":
        input_tensor, dim, index, mask, value = inputs
        return torch.gather(input_tensor, dim, index).masked_fill(mask, value)
    raise ValueError(f"unsupported KernelBench-X op: {op}")


def make_inputs(task: dict[str, Any], test: dict[str, Any], base_seed: int, device: str):
    import torch
    if is_kbx_task(task):
        return kbx_make_inputs(task, test, base_seed, device)
    op_family = task["op_family"]
    variant = task.get("variant", task.get("op_name", ""))
    dtype_name = test.get("dtype", task.get("dtype", "fp16"))
    layout = test.get("layout", "contiguous")
    seed = base_seed + int(test.get("seed_offset", 0))
    shape = test.get("shape", task["shape"])
    if op_family == "pointwise":
        x = make_tensor_2d(shape, dtype_name, seed, layout, device)
        n = int(shape[1])
        if variant in {"bias_gelu", "bias_silu"}:
            return x, make_vector(n, dtype_name, seed + 17, device)
        if variant == "residual_relu":
            residual = make_tensor_2d(shape, dtype_name, seed + 17, layout, device)
            return x, residual
        if variant == "broadcast_affine":
            return x, make_vector(n, dtype_name, seed + 17, device, scale=0.25), make_vector(n, dtype_name, seed + 23, device)
        if variant == "gated_silu_multiply":
            x2 = make_tensor_2d(shape, dtype_name, seed + 17, layout, device)
            return x, x2
        if variant == "clamp_mul_add":
            scale = make_vector(n, dtype_name, seed + 17, device, scale=0.25)
            bias = make_vector(n, dtype_name, seed + 23, device)
            return x, scale, bias, -2.0, 2.0
        raise ValueError(f"unsupported pointwise variant: {variant}")
    if op_family in {"reduction", "softmax", "layernorm"}:
        x = make_tensor_2d(shape, dtype_name, seed, layout, device)
        if op_family == "layernorm":
            n = int(shape[1])
            torch.manual_seed(seed + 17)
            gamma = torch.randn((n,), device=device, dtype=torch_dtype(dtype_name))
            beta = torch.randn((n,), device=device, dtype=torch_dtype(dtype_name))
            return x, gamma, beta, float(task.get("eps", 1e-5))
        if op_family == "reduction":
            return x, variant
        return (x,)
    if op_family == "matmul":
        dtype = torch_dtype(dtype_name)
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        torch.manual_seed(seed)
        a = torch.randn((m, k), device=device, dtype=dtype)
        torch.manual_seed(seed + 1)
        b = torch.randn((k, n), device=device, dtype=dtype)
        return a, b
    if op_family == "layout":
        dtype = torch_dtype(dtype_name)
        torch.manual_seed(seed)
        if variant == "transpose_copy":
            m, n = int(shape["M"]), int(shape["N"])
            if layout == "non_contiguous":
                x = torch.randn((n, m), device=device, dtype=dtype).t()
            else:
                x = torch.randn((m, n), device=device, dtype=dtype)
            return (x,)
        if variant == "gather_rows":
            rows, width, out_rows = int(shape["rows"]), int(shape["width"]), int(shape["out_rows"])
            x = torch.randn((rows, width), device=device, dtype=dtype)
            index_dtype = torch.int32 if layout == "index_int32" else torch.int64
            indices = torch.randint(0, rows, (out_rows,), device=device, dtype=index_dtype)
            return x, indices
        if variant == "embedding_lookup":
            vocab, dim = int(shape["vocab"]), int(shape["dim"])
            batch, seq = int(shape["batch"]), int(shape["seq"])
            weight = torch.randn((vocab, dim), device=device, dtype=dtype)
            index_dtype = torch.int32 if layout == "index_int32" else torch.int64
            indices = torch.randint(0, vocab, (batch, seq), device=device, dtype=index_dtype)
            return weight, indices
        if variant == "scatter_add":
            nnz, width, out_rows = int(shape["nnz"]), int(shape["width"]), int(shape["out_rows"])
            src = torch.randn((nnz, width), device=device, dtype=dtype)
            high = max(1, out_rows // 8) if layout == "collision_heavy" else out_rows
            indices = torch.randint(0, high, (nnz,), device=device, dtype=torch.int64)
            return src, indices, out_rows
        if variant == "slice_concat":
            b, n, cut = int(shape["B"]), int(shape["N"]), int(shape["cut"])
            if layout == "non_contiguous":
                a = torch.randn((n, b), device=device, dtype=dtype).t()
                other = torch.randn((n, b), device=device, dtype=dtype).t()
            else:
                a = torch.randn((b, n), device=device, dtype=dtype)
                other = torch.randn((b, n), device=device, dtype=dtype)
            return a, other, cut
        if variant == "strided_copy":
            return (make_strided_2d(int(shape["M"]), int(shape["N"]), dtype_name, seed, device, int(shape.get("row_stride", 2))),)
        raise ValueError(f"unsupported layout variant: {variant}")
    raise ValueError(f"unsupported op_family: {op_family}")


def reference(task: dict[str, Any], inputs: tuple[Any, ...]):
    import torch
    if is_kbx_task(task):
        return kbx_reference(task, inputs)
    op_family = task["op_family"]
    variant = task.get("variant", task.get("op_name", ""))
    if op_family == "pointwise":
        import torch.nn.functional as F
        if variant == "bias_gelu":
            x, bias = inputs
            return F.gelu(x.float() + bias.float()).to(x.dtype)
        if variant == "bias_silu":
            x, bias = inputs
            return F.silu(x.float() + bias.float()).to(x.dtype)
        if variant == "residual_relu":
            x, residual = inputs
            return torch.relu(x.float() + residual.float()).to(x.dtype)
        if variant == "broadcast_affine":
            x, scale, bias = inputs
            return (x.float() * scale.float() + bias.float()).to(x.dtype)
        if variant == "gated_silu_multiply":
            x1, x2 = inputs
            return (F.silu(x1.float()) * x2.float()).to(x1.dtype)
        if variant == "clamp_mul_add":
            x, scale, bias, min_value, max_value = inputs
            return (torch.clamp(x.float(), float(min_value), float(max_value)) * scale.float() + bias.float()).to(x.dtype)
        raise ValueError(f"unsupported pointwise variant: {variant}")
    if op_family == "reduction":
        x, reduce_op = inputs
        if reduce_op == "sum":
            return x.float().sum(dim=-1)
        if reduce_op == "max":
            return x.max(dim=-1).values
        raise ValueError(f"unsupported reduction op: {reduce_op}")
    if op_family == "softmax":
        (x,) = inputs
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
    if op_family == "layernorm":
        x, gamma, beta, eps = inputs
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        y = (xf - mean) / torch.sqrt(var + eps)
        return (y * gamma.float() + beta.float()).to(x.dtype)
    if op_family == "matmul":
        a, b = inputs
        return torch.matmul(a, b)
    if op_family == "layout":
        if variant == "transpose_copy":
            (x,) = inputs
            return x.t().contiguous()
        if variant == "gather_rows":
            x, indices = inputs
            return x[indices.long()]
        if variant == "embedding_lookup":
            weight, indices = inputs
            return weight[indices.long()]
        if variant == "scatter_add":
            src, indices, out_rows = inputs
            out = torch.zeros((int(out_rows), src.shape[1]), device=src.device, dtype=src.dtype)
            return out.index_add(0, indices.long(), src)
        if variant == "slice_concat":
            a, b, cut = inputs
            return torch.cat((a[:, : int(cut)], b[:, int(cut):]), dim=-1).contiguous()
        if variant == "strided_copy":
            (x,) = inputs
            return x.contiguous()
        raise ValueError(f"unsupported layout variant: {variant}")
    raise ValueError(f"unsupported op_family: {op_family}")


def compare_outputs(actual: Any, expected: Any, dtype_name: str, op_family: str) -> dict[str, Any]:
    import torch
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            return {"status": "fail", "reason": f"container mismatch actual={type(actual).__name__} expected={type(expected).__name__}"}
        child_results = [compare_outputs(a, e, dtype_name, op_family) for a, e in zip(actual, expected)]
        failed = [item for item in child_results if item.get("status") != "pass"]
        if failed:
            return {"status": "fail", "children": child_results, "reason": failed[0].get("reason", "nested output mismatch")}
        max_abs = max((float(item.get("max_abs_error", 0.0) or 0.0) for item in child_results), default=0.0)
        max_rel = max((float(item.get("max_rel_error", 0.0) or 0.0) for item in child_results), default=0.0)
        return {"status": "pass", "children": child_results, "max_abs_error": max_abs, "max_rel_error": max_rel}
    if not torch.is_tensor(expected):
        if torch.is_tensor(actual) and actual.numel() == 1:
            actual = actual.item()
        ok = actual == expected
        return {"status": "pass" if ok else "fail", "reason": "" if ok else f"value mismatch actual={actual} expected={expected}", "max_abs_error": 0.0 if ok else math.nan, "max_rel_error": 0.0 if ok else math.nan}
    if not torch.is_tensor(actual):
        return {"status": "fail", "reason": "candidate did not return a torch.Tensor"}
    if actual.shape != expected.shape:
        return {"status": "fail", "reason": f"shape mismatch actual={tuple(actual.shape)} expected={tuple(expected.shape)}"}
    if not (actual.is_floating_point() or expected.is_floating_point()):
        ok = torch.equal(actual, expected)
        mismatch = int((actual != expected).sum().item()) if actual.numel() else 0
        return {"status": "pass" if bool(ok) else "fail", "reason": "" if bool(ok) else f"integer mismatch_count={mismatch}", "max_abs_error": float(mismatch), "max_rel_error": 0.0}
    atol, rtol = tolerance(dtype_name, op_family)
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=True)
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return {"status": "pass" if bool(ok) else "fail", "atol": atol, "rtol": rtol, "max_abs_error": max_abs, "max_rel_error": max_rel}


def run_suite(
    task: dict[str, Any],
    hidden: dict[str, Any],
    fn: Callable[..., Any],
    base_seed: int,
    device: str,
    log_lines: list[str],
    suite_names: list[str] | None = None,
) -> dict[str, Any]:
    import torch
    all_suites = {"smoke": task.get("public_tests", [])[:1], "quick": task.get("public_tests", []), "hidden": hidden.get("hidden_tests", [])}
    selected = suite_names or ["smoke", "quick", "hidden"]
    result: dict[str, Any] = {}
    for suite_name in selected:
        tests = all_suites[suite_name]
        suite_records = []
        suite_ok = True
        for idx, test in enumerate(tests):
            try:
                inputs = make_inputs(task, test, base_seed + idx * 1000, device)
                if is_kbx_task(task):
                    expected = reference(task, clone_input_tree(inputs))
                    actual = fn(*clone_input_tree(inputs))
                else:
                    expected = reference(task, inputs)
                    actual = fn(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                cmp = compare_outputs(actual, expected, test.get("dtype", task.get("dtype", "fp16")), task["op_family"])
                record = {"test": test, **cmp}
                if cmp["status"] != "pass":
                    suite_ok = False
            except Exception as exc:
                suite_ok = False
                record = {"test": test, "status": "fail", "reason": str(exc), "traceback": traceback.format_exc(limit=8)}
            suite_records.append(record)
            if not suite_ok and suite_name == "smoke":
                break
        result[suite_name] = {"status": "pass" if suite_ok else "fail", "tests": suite_records}
        log_lines.append(f"{suite_name}: {result[suite_name]['status']}")
        if not suite_ok:
            break
    return result


def time_callable(fn: Callable[[], Any], warmup: int, repeats: int, device: str) -> dict[str, Any]:
    import torch
    samples: list[float] = []
    if device == "cuda":
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)))
    else:
        for _ in range(warmup):
            fn()
        for _ in range(repeats):
            start = time.perf_counter(); fn(); samples.append((time.perf_counter() - start) * 1000.0)
    samples = sorted(samples)
    p50 = statistics.median(samples) if samples else math.nan
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))] if samples else math.nan
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": statistics.mean(samples) if samples else math.nan, "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0, "min_ms": min(samples) if samples else math.nan, "samples": samples}


def benchmark(task: dict[str, Any], fn: Callable[..., Any], seed: int, device: str, warmup: int, repeats: int) -> dict[str, Any]:
    import torch
    test = task.get("public_tests", [])[0]
    inputs = make_inputs(task, test, seed, device)
    if is_kbx_task(task):
        candidate_t = time_callable(lambda: fn(*clone_input_tree(inputs)), warmup, repeats, device)
        eager_t = time_callable(lambda: reference(task, clone_input_tree(inputs)), warmup, repeats, device)
    else:
        candidate_t = time_callable(lambda: fn(*inputs), warmup, repeats, device)
        eager_t = time_callable(lambda: reference(task, inputs), warmup, repeats, device)
    compile_t: dict[str, Any] = {"status": "not_run"}
    try:
        compiled_ref = torch.compile(lambda *args: reference(task, args), fullgraph=False)
        if is_kbx_task(task):
            compile_t = {"status": "pass", **time_callable(lambda: compiled_ref(*clone_input_tree(inputs)), max(5, warmup // 5), repeats, device)}
        else:
            compile_t = {"status": "pass", **time_callable(lambda: compiled_ref(*inputs), max(5, warmup // 5), repeats, device)}
    except Exception as exc:
        compile_t = {"status": "fail", "reason": str(exc)}
    cand_p50 = float(candidate_t["p50_ms"])
    eager_p50 = float(eager_t["p50_ms"])
    compile_p50 = float(compile_t.get("p50_ms", math.nan)) if compile_t.get("status") == "pass" else math.nan
    return {"measurement": {"warmup": warmup, "repeats": repeats}, "candidate": candidate_t, "eager": eager_t, "torch_compile": compile_t, "speedup_vs_eager_p50": eager_p50 / cand_p50 if cand_p50 > 0 else 0.0, "speedup_vs_torch_compile_p50": compile_p50 / cand_p50 if cand_p50 > 0 and not math.isnan(compile_p50) else 0.0}


def profile_summary(task: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    symptom = "compute_bound" if task.get("op_family") == "matmul" else "memory_bound"
    speedup = bench.get("speedup_vs_eager_p50", 0.0) if bench else 0.0
    actions = ["verify_p95_stability", "try_small_knob_search_only"] if speedup >= 1.0 else ["inspect_memory_access_pattern", "reduce_register_pressure", "retune_tile_or_block_size"]
    return {"dominant_symptom": symptom, "evidence": {"heuristic": "op-family fallback summary; replace with NCU/Triton profiler when available", "speedup_vs_eager_p50": speedup}, "candidate_actions": actions}


def status_at(data: dict[str, Any], path: str, default: str = "not_run") -> str:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return str(cur)


def print_summary(results: dict[str, Any], out_dir: Path) -> None:
    print("H20 harness summary")
    print(f"compile: {status_at(results, 'compile.status')}")
    print(
        "correctness: "
        f"smoke={status_at(results, 'correctness.smoke.status')} "
        f"quick={status_at(results, 'correctness.quick.status')} "
        f"hidden={status_at(results, 'correctness.hidden.status')}"
    )
    benchmark = results.get("benchmark", {})
    if benchmark:
        print(
            "performance: "
            f"p50_ms={float(benchmark.get('latency_p50_ms', math.nan)):.6g} "
            f"p95_ms={float(benchmark.get('latency_p95_ms', math.nan)):.6g} "
            f"speedup_vs_eager={float(benchmark.get('speedup_vs_eager_p50', 0.0)):.4g} "
            f"speedup_vs_torch_compile={float(benchmark.get('speedup_vs_torch_compile_p50', 0.0)):.4g}"
        )
    else:
        print("performance: not_run")
    diagnosis = results.get("diagnosis") or status_at(results, "compile.reason", "")
    if diagnosis:
        print(f"diagnosis: {diagnosis}")
    print("outputs:")
    for name in ("results.json", "compile.log", "correctness.log", "benchmark.json", "profile_summary.json"):
        print(f"  {name}: {out_dir / name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["full", "compile", "smoke", "quick", "hidden", "benchmark"], default="full")
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-tests", default=None)
    parser.add_argument("--run", type=int, default=None, help="Run index used for reproducible input generation.")
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true", help="Fail instead of silently falling back to CPU when CUDA is unavailable.")
    args = parser.parse_args()
    if args.run is None:
        args.run = 0 if args.seed is None else args.seed
    args.seed = args.run

    import torch
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    task = read_json(args.task)
    hidden = read_json(args.hidden_tests, default={"hidden_tests": []}) if args.hidden_tests else {"hidden_tests": []}
    candidate_path = Path(args.candidate)
    warmup = int(args.warmup if args.warmup is not None else task.get("budget", {}).get("warmup", 100))
    repeats = int(args.repeats if args.repeats is not None else task.get("budget", {}).get("repeats", 500))
    requested_device = args.device
    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available and args.require_cuda:
        results = {
            "schema_version": "0.1",
            "run_id": f"harness_{short_hash([task, args.seed, args.stage, sha256_file(candidate_path), 'cuda_unavailable'])}",
            "stage": args.stage,
            "timestamp": utc_now(),
            "task_id": task.get("task_id"),
            "op_family": task.get("op_family"),
            "candidate_hash": sha256_file(candidate_path),
            "compile": {"status": "fail", "reason": "CUDA requested but torch.cuda.is_available() is false"},
            "anti_cheating": {"status": "not_run", "judge": "llm_judge_pending", "issues": []},
            "correctness": {},
            "benchmark": {},
            "profile_summary": {},
            "cost": {"iterations": 1, "gpu_benchmark_runs": 0, "wall_time_s": 0},
            "runtime": {
                "requested_device": requested_device,
                "actual_device": "none",
                "cuda_available": False,
                "cuda_device_count": int(torch.cuda.device_count()),
            },
            "final_decision": "FAIL",
            "diagnosis": "CUDA requested but unavailable; H20 benchmark was not run.",
        }
        write_json(out_dir / "results.json", results)
        (out_dir / "compile.log").write_text(results["diagnosis"] + "\n", encoding="utf-8")
        (out_dir / "correctness.log").write_text("", encoding="utf-8")
        write_json(out_dir / "benchmark.json", {})
        write_json(out_dir / "profile_summary.json", {})
        print_summary(results, out_dir)
        return 1
    device = requested_device if requested_device != "cuda" or cuda_available else "cpu"
    started = time.time()
    compile_log: list[str] = []
    correctness_log: list[str] = []
    results: dict[str, Any] = {"schema_version": "0.1", "run_id": f"harness_{short_hash([task, args.seed, args.stage, sha256_file(candidate_path)])}", "stage": args.stage, "timestamp": utc_now(), "task_id": task.get("task_id"), "op_family": task.get("op_family"), "candidate_hash": sha256_file(candidate_path), "compile": {"status": "not_run"}, "anti_cheating": {"status": "not_run", "judge": "llm_judge_pending", "issues": []}, "correctness": {}, "benchmark": {}, "profile_summary": {}, "cost": {"iterations": 1, "gpu_benchmark_runs": 0, "wall_time_s": 0}, "runtime": {"requested_device": requested_device, "actual_device": device, "cuda_available": bool(cuda_available), "cuda_device_count": int(torch.cuda.device_count())}}
    try:
        entrypoint = task.get("kernelbenchx", {}).get("entrypoint") if is_kbx_task(task) else None
        fn = load_candidate(candidate_path, str(entrypoint) if entrypoint else None)
        results["compile"] = {"status": "pass"}; compile_log.append("candidate import: pass")
    except Exception as exc:
        results["compile"] = {"status": "fail", "reason": str(exc), "traceback": traceback.format_exc(limit=12)}
        results["final_decision"] = "FAIL"; results["cost"]["wall_time_s"] = time.time() - started
        write_json(out_dir / "results.json", results); (out_dir / "compile.log").write_text(traceback.format_exc(), encoding="utf-8")
        (out_dir / "correctness.log").write_text("", encoding="utf-8"); write_json(out_dir / "benchmark.json", {}); write_json(out_dir / "profile_summary.json", {})
        print_summary(results, out_dir); return 1

    if args.stage == "compile":
        results["final_decision"] = "COMPILE_PASS"
        results["cost"]["wall_time_s"] = time.time() - started
        write_json(out_dir / "results.json", results); write_json(out_dir / "benchmark.json", {}); write_json(out_dir / "profile_summary.json", {})
        (out_dir / "compile.log").write_text("\n".join(compile_log) + "\n", encoding="utf-8")
        (out_dir / "correctness.log").write_text("", encoding="utf-8")
        print_summary(results, out_dir)
        return 0

    suite_names = ["smoke", "quick", "hidden"] if args.stage == "full" else (["hidden"] if args.stage == "benchmark" else [args.stage])
    correctness = run_suite(task, hidden, fn, args.seed, device, correctness_log, suite_names=suite_names)
    results["correctness"] = correctness
    required_suite = "hidden" if args.stage in {"full", "benchmark"} else args.stage
    correctness_ok = str(correctness.get(required_suite, {}).get("status", "fail")) == "pass"
    failed_suite = next((name for name in suite_names if str(correctness.get(name, {}).get("status", "fail")) != "pass"), required_suite)
    hidden_ok = str(correctness.get("hidden", {}).get("status", "fail")) == "pass"
    bench: dict[str, Any] = {}; prof: dict[str, Any] = {}
    if args.stage in {"full", "benchmark"} and hidden_ok:
        bench = benchmark(task, fn, args.seed, device, warmup, repeats)
        prof = profile_summary(task, bench)
        results["benchmark"] = {"latency_p50_ms": bench["candidate"]["p50_ms"], "latency_p95_ms": bench["candidate"]["p95_ms"], "latency_mean_ms": bench["candidate"]["mean_ms"], "latency_std_ms": bench["candidate"]["std_ms"], "speedup_vs_eager_p50": bench.get("speedup_vs_eager_p50", 0.0), "speedup_vs_torch_compile_p50": bench.get("speedup_vs_torch_compile_p50", 0.0)}
        results["profile_summary"] = prof; results["cost"]["gpu_benchmark_runs"] = repeats
        results["final_decision"] = "KEEP" if results["benchmark"]["speedup_vs_eager_p50"] >= 1.0 else "DISCARD"
    elif correctness_ok:
        results["final_decision"] = "PASS"
    else:
        results["final_decision"] = "FAIL"
        results["diagnosis"] = f"{failed_suite} correctness failed"
    results["cost"]["wall_time_s"] = time.time() - started
    write_json(out_dir / "results.json", results); write_json(out_dir / "benchmark.json", bench); write_json(out_dir / "profile_summary.json", prof)
    (out_dir / "compile.log").write_text("\n".join(compile_log) + "\n", encoding="utf-8")
    (out_dir / "correctness.log").write_text("\n".join(correctness_log) + "\n", encoding="utf-8")
    print_summary(results, out_dir)
    return 0 if results["final_decision"] in {"KEEP", "DISCARD", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
