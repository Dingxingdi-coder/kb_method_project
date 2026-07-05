"""Candidate kernel for row reduction. Interface: candidate(x, reduce_op) -> y. Do not call torch.sum or torch.max."""

def candidate(x, reduce_op):
    raise NotImplementedError("agent must implement row reduction")
