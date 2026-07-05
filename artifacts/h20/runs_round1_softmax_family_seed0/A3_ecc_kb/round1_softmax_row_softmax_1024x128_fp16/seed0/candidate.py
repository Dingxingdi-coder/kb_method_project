"""Candidate kernel for row softmax. Interface: candidate(x) -> y."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    vals = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    vals = vals - tl.max(vals, axis=0)
    numer = tl.exp(vals)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(y_ptr + row * n_cols + offsets, out, mask=mask)


def _warps_for(block_size):
    if block_size <= 256:
        return 4
    if block_size <= 2048:
        return 8
    return 16


def candidate(x):
    x_work = x if x.is_contiguous() else x.contiguous()
    n_cols = x_work.shape[-1]
    n_rows = x_work.numel() // n_cols
    y = torch.empty_like(x_work)
    block_size = triton.next_power_of_2(n_cols)
    _row_softmax_kernel[(n_rows,)](
        x_work, y, n_cols, block_size, num_warps=_warps_for(block_size)
    )
    return y
