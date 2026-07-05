"""Candidate kernel for row softmax. Interface: candidate(x) -> y. Do not call torch.softmax."""

def candidate(x):
    raise NotImplementedError("agent must implement row softmax")
