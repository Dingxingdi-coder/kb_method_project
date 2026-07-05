"""Candidate kernel for matmul. Interface: candidate(a, b) -> c. Do not call torch.matmul."""

def candidate(a, b):
    raise NotImplementedError("agent must implement matmul")
