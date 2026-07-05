"""Candidate kernel for LayerNorm forward. Interface: candidate(x, gamma, beta, eps) -> y."""

def candidate(x, gamma, beta, eps):
    raise NotImplementedError("agent must implement layernorm forward")
