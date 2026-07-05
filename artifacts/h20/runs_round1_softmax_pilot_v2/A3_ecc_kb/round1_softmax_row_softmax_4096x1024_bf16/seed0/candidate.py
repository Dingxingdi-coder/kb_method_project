"""Candidate kernel for row softmax. Interface: candidate(x) -> y."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < n_cols
    row_start = row * n_cols

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def candidate(x):
    if x.numel() == 0:
        return torch.empty_like(x)

    x_work = x if x.is_contiguous() else x.contiguous()
    n_cols = x_work.shape[-1]
    if n_cols == 0:
        return torch.empty_like(x_work)

    rows = x_work.numel() // n_cols
    y = torch.empty_like(x_work)
    block = _next_power_of_2(n_cols)
    num_warps = 4
    if block >= 4096:
        num_warps = 8

    _row_softmax_kernel[(rows,)](x_work, y, n_cols, block, num_warps=num_warps)
    return y.reshape(x_work.shape)
