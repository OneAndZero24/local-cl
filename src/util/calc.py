import torch


def compute_mean_and_var(tensors, alpha=0.1):
    """
    Compute the mean and variance of a list of tensors.

    Args:
        tensors (list of torch.Tensor): A list of tensors to compute the mean and variance for.
        alpha (float, optional): A parameter that is currently not used in the function. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the mean of means and the mean of variances of the input tensors.
    """


    means = torch.stack([tensor.mean(dim=-1) for tensor in tensors])
    variances = torch.stack([tensor.var(dim=-1) for tensor in tensors])

    return means.mean(dim=0), variances.mean(dim=0)