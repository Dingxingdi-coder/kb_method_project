"""Candidate kernel for row softmax. Interface: candidate(x) -> y."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    values = tl.load(x_ptr + base, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numer = tl.exp(values)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(y_ptr + base, out, mask=mask)


def _launch_config(n_cols):
    block = triton.next_power_of_2(n_cols)
    if block >= 2048:
        num_warps = 8
    elif block >= 512:
        num_warps = 4
    else:
        num_warps = 1
    return block, num_warps


def candidate(x):
    if not x.is_cuda:
        raise RuntimeError("candidate expects a CUDA tensor")

    x_work = x.contiguous() if not x.is_contiguous() else x
    y = torch.empty_like(x_work)

    n_cols = x_work.shape[-1]
    if n_cols == 0 or x_work.numel() == 0:
        return y.reshape(x_work.shape)

    rows = x_work.numel() // n_cols
    block, num_warps = _launch_config(n_cols)
    _row_softmax_kernel[(rows,)](
        x_work,
        y,
        n_cols,
        block,
        num_warps=num_warps,
        num_stages=4,
    )
    return y.reshape(x_work.shape)
