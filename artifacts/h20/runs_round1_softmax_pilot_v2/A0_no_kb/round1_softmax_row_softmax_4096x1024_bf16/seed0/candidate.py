"""Candidate kernel for row softmax. Interface: candidate(x) -> y."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols:tl.constexpr, block:tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, block)
    mask = offs < n_cols
    base = row * n_cols + offs

    vals = tl.load(x_ptr + base, mask=mask, other=-float("inf")).to(tl.float32)
    vals = vals - tl.max(vals, axis=0)
    numer = tl.exp(vals)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(y_ptr + base, out, mask=mask)

def candidate(x):
    if x.numel() == 0:
        return torch.empty_like(x)

    n_cols = x.shape[-1]
    if n_cols == 0:
        return torch.empty_like(x)

    x_work = x if x.is_contiguous() else x.contiguous()
    y = torch.empty_like(x_work)
    n_rows = x_work.numel() // n_cols

    block = triton.next_power_of_2(n_cols)
    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16

    _row_softmax_kernel[(n_rows,)](x_work, y, n_cols, block, num_warps=num_warps)
    return y.reshape(x.shape)
