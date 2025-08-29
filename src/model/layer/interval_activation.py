import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import wandb


def _gaussian(x, alpha=0.1):
    return torch.exp(-alpha*x**2)

def _relu_hill(x):
    return torch.where(
        x < -1, torch.zeros_like(x),
        torch.where(
            x < 0, x + 1,
            torch.where(
                x < 1, 1 - x,
                torch.zeros_like(x)
            )
        )
    )

def _hard_bound(lower_bound, upper_bound, x):
    return torch.where((lower_bound < x) & (x < upper_bound), torch.zeros_like(x), x)

def _soft_bound(lower_bound, upper_bound, x):
    middle = (lower_bound + upper_bound) / 2
    return torch.where((lower_bound < x) & (x < upper_bound), (x - middle)**2, x)

def no_bound():
    def _no_bound(lower_bound, upper_bound, x):
        return x

class IntervalActivation(nn.Module):
    """
    A neural network module that applies a custom activation function and bounds each element independently based on percentiles.

    Attributes:
        input_shape (tuple or int): Shape of the input tensor (flattened size).
        lower_percentile (float): Lower percentile for min bound calculation.
        upper_percentile (float): Upper percentile for max bound calculation.
        act_function (callable): Activation function applied to each element.
        test_act_buffer (list): Stores output samples for percentile calculation.
        curr_task_act_buffer (list): Stores output samples for the regularization term.
        min (torch.Tensor): Minimum bounds for each element.
        max (torch.Tensor): Maximum bounds for each element.

    Methods:
        reset_range():
            Computes min and max bounds for each element from the buffer using percentiles.
        forward(x):
            Applies the activation function and bounds to each element independently, updates buffer.
    """

    def __init__(self,
        input_shape: tuple,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        act_function: callable = _gaussian,
        log_name: str = None,
    ):
        """
        Initializes the IntervalActivation layer.

        Args:
            input_shape (tuple): Shape of the input tensor.
            lower_percentile (float, optional): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float, optional): Upper percentile for max bound. Defaults to 0.95.
            act_function (callable, optional): Activation function. Defaults to _gaussian.
            name (str, optional): Name of the layer for wandb logging. Defaults to None.
        """

        super().__init__()
        self.input_shape = np.prod(input_shape)
        self.act_function = act_function
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        self.curr_task_act_buffer = [] # samples x input_shape flattened
        self.test_act_buffer = [] # activations from test set to calculate interval bounds
        self.min = torch.zeros(self.input_shape)
        self.max = torch.zeros(self.input_shape)
        self.log_name = log_name

    def reset_range(self):
        """
        Compute per-feature percentiles from collected test_act_buffer and update min/max.
        After computing, clears test_act_buffer.
        """
        if len(self.test_act_buffer) == 0:
            return

        activations = torch.cat(self.test_act_buffer, dim=0)
        device = activations.device
        activations = activations.to(device)

        sorted_buf, _ = torch.sort(activations, dim=0)

        n = sorted_buf.size(0)
        if n == 0:
            return

        l_idx = int(np.clip(int(n * self.lower_percentile), 0, n - 1))
        u_idx = int(np.clip(int(n * self.upper_percentile), 0, n - 1))

        min_vals = sorted_buf[l_idx]
        max_vals = sorted_buf[u_idx]

        min_vals = min_vals.to(device)
        max_vals = max_vals.to(device)

        if self.min is None or self.max is None:
            self.min = min_vals.clone().detach()
            self.max = max_vals.clone().detach()
        else:
            self.min = torch.minimum(self.min.to(device), min_vals)
            self.max = torch.maximum(self.max.to(device), max_vals)

        self.test_act_buffer = []

        if self.log_name is not None and wandb.run is not None:
            interval_size = (self.max - self.min).cpu()
            for i in range(self.input_shape):
                prefix = f"{self.log_name}/neuron_{i}"
                wandb.log({
                    f"{prefix}/min": float(self.min[i].cpu().item()),
                    f"{prefix}/max": float(self.max[i].cpu().item()),
                    f"{prefix}/interval_size": float(interval_size[i].item()),
                })



    def forward(self, x):
        """
        Applies the activation function and bounds to each element independently, updates buffer.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with activation and bounds applied elementwise.
        """
        x_flat = x.view(x.shape[0], -1)

        # TODO: Change this
        activated = 5*F.tanh(x_flat)

        if (self.min.sum() == 0 and self.max.sum() == 0):
            output = activated
        else:
            lb = self.min.unsqueeze(0).expand_as(x_flat)
            ub = self.max.unsqueeze(0).expand_as(x_flat)

            mask = ((x_flat < lb) | (x_flat > ub)).float()
            output = activated * mask + activated.detach() * (1 - mask)

        if self.training:
            self.test_act_buffer.extend(list(output.detach().cpu()))
        self.curr_task_act_buffer.extend(list(output.detach().cpu()))

        return output.view_as(x)