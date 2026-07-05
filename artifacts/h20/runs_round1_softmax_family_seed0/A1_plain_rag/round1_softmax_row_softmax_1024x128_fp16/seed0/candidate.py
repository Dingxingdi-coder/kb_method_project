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

    vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    vals = vals - tl.max(vals, axis=0)
    numer = tl.exp(vals)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(y_ptr + row_start + offsets, out, mask=mask)


def _launch_config(n_cols):
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return block_size, num_warps


def candidate(x):
    x_work = x if x.is_contiguous() else x.contiguous()
    n_cols = x_work.shape[-1]
    n_rows = x_work.numel() // n_cols
    y = torch.empty_like(x_work)
    block_size, num_warps = _launch_config(n_cols)
    _row_softmax_kernel[(n_rows,)](x_work, y, n_cols, BLOCK_SIZE=block_size, num_warps=num_warps)
    return y
