"""Candidate kernel for row softmax. Interface: candidate(x) -> y. Do not call torch.softmax."""

import torch

def candidate(x):
    work = x.to(torch.float32)
    shifted = work - work.amax(dim=-1, keepdim=True)
    exp_x = torch.exp(shifted)
    y = exp_x / exp_x.sum(dim=-1, keepdim=True)
    return y.to(dtype=x.dtype)
