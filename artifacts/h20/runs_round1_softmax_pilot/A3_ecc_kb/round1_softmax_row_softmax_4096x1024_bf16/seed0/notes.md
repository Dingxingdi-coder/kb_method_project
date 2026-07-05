# Agent notes

- Implemented stable row softmax over the last dimension without calling torch.softmax.
- Reductions and exponentiation use fp32, then the result is cast back to the input dtype.
