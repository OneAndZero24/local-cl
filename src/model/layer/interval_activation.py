import torch
from torch import nn
import numpy as np

import wandb


def _gaussian(x):
    return torch.exp(-x**2)

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

class IntervalActivation(nn.Module):
    """
    A neural network module that applies a custom activation function and bounds each element independently based on percentiles.

    Attributes:
        input_shape (tuple or int): Shape of the input tensor (flattened size).
        lower_percentile (float): Lower percentile for min bound calculation.
        upper_percentile (float): Upper percentile for max bound calculation.
        act_function (callable): Activation function applied to each element.
        bound_multiplier (callable): Function to apply bounds to each element.
        buffer (list): Stores output samples for percentile calculation.
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
        bound_multiplier: callable = _hard_bound,
        log_name: str = None,
    ):
        """
        Initializes the IntervalActivation layer.

        Args:
            input_shape (tuple): Shape of the input tensor.
            lower_percentile (float, optional): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float, optional): Upper percentile for max bound. Defaults to 0.95.
            act_function (callable, optional): Activation function. Defaults to _gaussian.
            bound_multiplier (callable, optional): Function to apply bounds. Defaults to identity.
            name (str, optional): Name of the layer for wandb logging. Defaults to None.
        """

        super().__init__()
        self.input_shape = np.prod(input_shape)
        self.act_function = act_function
        self.bound_multiplier = bound_multiplier if bound_multiplier is not None else lambda lower_bound, upper_bound, x: x
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        self.buffer = [] # samples x input_shape flattened
        self.min = torch.zeros(self.input_shape)
        self.max = torch.zeros(self.input_shape)
        self.dummy_range = True
        self.log_name = log_name

    def reset_range(self):
        """
        Computes min and max bounds for each element from the buffer using percentiles.
        Resets the buffer after calculation.
        """

        if self.log_name is not None and wandb.run is not None:
            interval_size = self.max - self.min
            for i in range(self.input_shape):
                prefix = f"{self.log_name}/neuron_{i}"
                wandb.log({
                    f"{prefix}/min": self.min[i].item(),
                    f"{prefix}/max": self.max[i].item(),
                    f"{prefix}/interval_size": interval_size[i].item(),
                })

        if len(self.buffer) > 0:
            transposed = [[] for _ in range(len(self.input_shape))] # input_shape flattened x samples
            for sample in self.buffer:
                for i, val in enumerate(sample):
                    transposed[i].append(val)
            min_vals = []
            max_vals = []
            for element in transposed:
                sorted_buf = sorted(element)
                l_idx = int(len(sorted_buf) * self.lower_percentile)
                u_idx = int(len(sorted_buf) * self.upper_percentile)
                min_vals.append(sorted_buf[l_idx])
                max_vals.append(sorted_buf[u_idx])
            self.min = torch.minimum(self.min, torch.tensor(min_vals))
            self.max = torch.maximum(self.max, torch.tensor(max_vals))
            self.dummy_range = False
        self.buffer = []



    def forward(self, x):
        """
        Applies the activation function and bounds to each element independently, updates buffer.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with activation and bounds applied elementwise.
        """
        x_flat = x.view(-1)
        output_flat = self.act_function(x_flat)
        output = self.bound_multiplier(
            self.min.repeat(x.shape[0]), 
            self.max.repeat(x.shape[0]), 
            output_flat
        )
        self.buffer.append(output.detach().cpu())
        return output.view(x.shape)