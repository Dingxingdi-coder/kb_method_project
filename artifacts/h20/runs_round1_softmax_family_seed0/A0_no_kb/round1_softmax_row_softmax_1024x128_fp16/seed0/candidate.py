"""Candidate kernel for row softmax. Interface: candidate(x) -> y."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = row_id * n_cols + cols

    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numer = tl.exp(values)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(y_ptr + offsets, out, mask=mask)


def _num_warps(block_size):
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1


def candidate(x):
    x_work = x if x.is_contiguous() else x.contiguous()
    n_cols = x_work.shape[-1]
    n_rows = x_work.numel() // n_cols
    y = torch.empty_like(x_work)
    block_size = triton.next_power_of_2(n_cols)
    _row_softmax_kernel[(n_rows,)](
        x_work,
        y,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return y
