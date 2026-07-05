# Agent notes

- Implemented row softmax along the last dimension with float32 max/exp/sum/divide and cast back to the input dtype.
- Kept the code shape-generic; no public-shape constants and no `torch.softmax` call.
