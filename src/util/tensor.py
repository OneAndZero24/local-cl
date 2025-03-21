from typing import Optional

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


def get_2D_classes_slice(tensor: torch.Tensor, classes: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Extracts a slice of a 2D tensor along the first dimension, using the specified classes.

    Args:
        tensor (torch.Tensor): The input tensor to be sliced.
        classes (torch.Tensor): The classes to be extracted from the tensor.
        
    Returns:
        torch.Tensor: A new tensor containing the specified classes from the input tensor.
    """

    if classes is None:
        return tensor[:classes, :]
    return tensor