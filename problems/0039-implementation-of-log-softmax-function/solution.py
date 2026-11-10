import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    maxx = max(scores)
    tnsr = torch.tensor(scores, dtype=torch.float32)
    exp_sum = torch.sum(torch.exp(tnsr - maxx), dim=0)

    return tnsr - maxx - torch.log(exp_sum)