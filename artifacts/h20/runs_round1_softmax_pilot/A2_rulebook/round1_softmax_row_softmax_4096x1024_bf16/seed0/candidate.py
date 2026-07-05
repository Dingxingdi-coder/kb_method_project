"""Candidate kernel for row softmax. Interface: candidate(x) -> y. Do not call torch.softmax."""

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only syntax/import fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_vals = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        x_vals = x_vals - tl.max(x_vals, axis=0)
        numer = tl.exp(x_vals)
        denom = tl.sum(numer, axis=0)
        y_vals = numer / denom
        tl.store(y_ptr + row * n_cols + offsets, y_vals, mask=mask)


def _fallback_softmax(x):
    if x.shape[-1] == 0:
        return torch.empty_like(x)
    x_fp32 = x.float()
    shifted = x_fp32 - torch.amax(x_fp32, dim=-1, keepdim=True)
    numer = torch.exp(shifted)
    return (numer / torch.sum(numer, dim=-1, keepdim=True)).to(dtype=x.dtype)


def candidate(x):
    if x.shape[-1] == 0:
        return torch.empty_like(x)
    if triton is None or not x.is_cuda:
        return _fallback_softmax(x)

    x_work = x.contiguous()
    y = torch.empty_like(x_work)
    n_cols = x_work.shape[-1]
    n_rows = x_work.numel() // n_cols
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 131072:
        return _fallback_softmax(x)

    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 8192:
        num_warps = 16

    _row_softmax_kernel[(n_rows,)](
        x_work,
        y,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y.reshape(x.shape)
