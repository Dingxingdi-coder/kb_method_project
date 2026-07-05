"""Candidate kernel for row softmax. Interface: candidate(x) -> y. Do not call torch.softmax."""

import torch


def candidate(x):
    work = x.to(torch.float32)
    shifted = work - work.amax(dim=-1, keepdim=True)
    numer = torch.exp(shifted)
    denom = numer.sum(dim=-1, keepdim=True)
    return (numer / denom).to(dtype=x.dtype)
