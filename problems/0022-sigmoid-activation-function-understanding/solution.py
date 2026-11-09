import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z_tensor = torch.tensor(z)
    print(z_tensor)
    print(z_tensor.dtype)
    print(z_tensor.shape)
    return (1 / (1 + torch.exp(-z_tensor))).item()