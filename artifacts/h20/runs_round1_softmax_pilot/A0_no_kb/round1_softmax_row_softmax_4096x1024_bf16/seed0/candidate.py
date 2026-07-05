"""Candidate kernel for row softmax. Interface: candidate(x) -> y. Do not call torch.softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < n_cols
    row_start = row * n_cols
    values = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numer = tl.exp(values)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(y_ptr + row_start + offsets, out, mask=mask)

def candidate(x):
    n_cols = x.shape[-1]
    if n_cols == 0:
        return torch.empty_like(x)

    if not x.is_cuda:
        values = x.float()
        values = values - values.max(dim=-1, keepdim=True).values
        numer = values.exp()
        return (numer / numer.sum(dim=-1, keepdim=True)).to(dtype=x.dtype)

    original_shape = x.shape
    x_2d = x.contiguous().view(-1, n_cols)
    y = torch.empty_like(x_2d)
    block = triton.next_power_of_2(n_cols)

    # Keep the single-program row reduction within Triton's practical block limit.
    if block > 131072:
        values = x.float()
        values = values - values.max(dim=-1, keepdim=True).values
        numer = values.exp()
        return (numer / numer.sum(dim=-1, keepdim=True)).to(dtype=x.dtype)

    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16

    _row_softmax_kernel[(x_2d.shape[0],)](x_2d, y, n_cols, block, num_warps=num_warps)
    return y.view(original_shape)
