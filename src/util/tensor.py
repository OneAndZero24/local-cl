import torch


def pad_zero_dim0(tensor: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """
    Pads a tensor with zeros along the first dimension to match the specified size.

    Args:
        tensor (torch.Tensor): The input tensor to be padded.
        size (torch.Size): The target size for the output tensor.
        
    Returns:
        torch.Tensor: A new tensor with the same content as the input tensor, 
                      but padded with zeros along the first dimension to match the specified size.
    """

    s = tensor.shape[0]
    tmp = torch.zeros(size).to(tensor.device)
    tmp[:s] = tensor
    return tmp