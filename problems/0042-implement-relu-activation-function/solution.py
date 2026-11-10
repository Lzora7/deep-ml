import torch

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    relu = torch.tensor(max(z, 0), dtype=torch.float32)
    return relu
