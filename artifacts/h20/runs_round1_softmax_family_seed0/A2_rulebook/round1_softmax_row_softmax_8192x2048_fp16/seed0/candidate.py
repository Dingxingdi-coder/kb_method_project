"""Candidate kernel for row softmax. Interface: candidate(x) -> y."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row * n_cols
    values = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(y_ptr + row_start + offsets, output, mask=mask)


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def candidate(x):
    x_work = x if x.is_contiguous() else x.contiguous()
    n_cols = x_work.shape[-1]
    y = torch.empty_like(x_work)
    n_rows = x_work.numel() // n_cols
    block_size = _next_power_of_2(n_cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    _row_softmax_kernel[(n_rows,)](x_work, y, n_cols, BLOCK_SIZE=block_size, num_warps=num_warps)
    return y
