import torch
import torch.nn as nn

def tensor_prompt(a: int, b: int, c: int=None, ortho: bool=False) -> nn.Parameter:
        """
        Creates a learnable torch parameter tensor with optional orthogonal or uniform initialization.

        Args:
            a (int): The size of the first dimension of the tensor.
            b (int): The size of the second dimension of the tensor.
            c (int, optional): The size of the third dimension of the tensor. If None, a 2D tensor is created. Defaults to None.
            ortho (bool, optional): If True, initializes the tensor with orthogonal initialization. If False, uses uniform initialization. Defaults to False.

        Returns:
            torch.nn.Parameter: A learnable parameter tensor of shape (a, b) or (a, b, c), initialized as specified.
        
        Source: https://github.com/GT-RIPL/CODA-Prompt/blob/main/models/zoo.py
        """
        if c is None:
            p = nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
        else:
            p = nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.uniform_(p)
        return p

def ortho_penalty(t: torch.Tensor) -> torch.Tensor:
    """
    Computes the orthogonality penalty for a given matrix.

    This function measures how close the input matrix `t` is to being orthogonal by calculating
    the mean squared difference between `t @ t.T` and the identity matrix. The penalty is lower
    when `t` is more orthogonal.

    Args:
        t (torch.Tensor): A 2D tensor to be evaluated for orthogonality. Should be on CUDA device.

    Returns:
        torch.Tensor: A scalar tensor representing the mean squared orthogonality penalty.
    """
    device = t.device
    return ((t @t.T - torch.eye(t.shape[0]).to(device))**2).mean()