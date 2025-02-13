import torch


def compute_ema_and_emv(tensors, alpha=0.1):
    """
    Compute the Exponential Moving Average (EMA) and Exponential Moving Variance (EMV) of a list of tensors.
    
    Args:
        tensors (list of torch.Tensor): A list of tensors for which the EMA and EMV are to be computed.
        alpha (float, optional): The smoothing factor for the EMA and EMV calculations. Default is 0.1.

    Returns:
        tuple: A tuple containing two tensors:
            - ema (torch.Tensor): The computed Exponential Moving Average.
            - emv (torch.Tensor): The computed Exponential Moving Variance.
    """

    ema = torch.zeros_like(tensors[0])
    emv = torch.zeros_like(tensors[0])

    for tensor in tensors:
        diff = tensor - ema
        ema = alpha * tensor + (1 - alpha) * ema
        emv = alpha * diff**2 + (1 - alpha) * emv
    
    return ema, emv