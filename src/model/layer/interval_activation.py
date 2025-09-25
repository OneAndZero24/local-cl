import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import wandb


class IntervalActivation(nn.Module):
    """
    IntervalActivation layer for preserving learned representations within a hypercube.

    This layer applies a Leaky ReLU activation and tracks the range of activations 
    across batches. It defines a [lb, ub] hypercube per neuron, which can be used 
    to enforce that activations within this cube remain unchanged when learning 
    new tasks.

    Attributes:
        input_shape (tuple or int): Flattened size of input tensor.
        lower_percentile (float): Lower percentile for min bound computation.
        upper_percentile (float): Upper percentile for max bound computation.
        test_act_buffer (list): Stores activations for percentile computation in eval mode.
        min (torch.Tensor): Lower bound per neuron (updated via reset_range).
        max (torch.Tensor): Upper bound per neuron (updated via reset_range).
        curr_task_last_batch (torch.Tensor): Stores last batch activations during training.

    Methods:
        reset_range():
            Computes per-feature min and max bounds using collected activations
            from test_act_buffer. Updates self.min and self.max.
        forward(x):
            Computes Leaky ReLU activation, saves batch activations and mask.
    """

    def __init__(self,
        input_shape: tuple,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        log_name: str = None,
    ) -> None:
        """
        Initializes the IntervalActivation layer.

        Args:
            input_shape (tuple): Shape of the input tensor.
            lower_percentile (float, optional): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float, optional): Upper percentile for max bound. Defaults to 0.95.
            log_name (str, optional): Name of the layer for wandb logging. Defaults to None.
        """

        super().__init__()
        self.input_shape = np.prod(input_shape)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        self.test_act_buffer = []
        self.min = None
        self.max = None
        self.log_name = log_name

        self.curr_task_last_batch = None

    def reset_range(self):
        """
        Updates the [min, max] hypercube for each neuron using collected activations.

        Steps:
            1. Concatenates stored activations in test_act_buffer.
            2. Sorts activations and selects lower and upper percentiles.
            3. Updates self.min and self.max by taking element-wise min/max.
            4. Clears the test_act_buffer.
            5. Optionally logs per-neuron min, max, and interval size to wandb.
        """
        
        if len(self.test_act_buffer) == 0:
            return

        activations = torch.stack(self.test_act_buffer, dim=0).to(self.test_act_buffer[0].device)  # shape: [n_samples, d]
        sorted_buf, _ = torch.sort(activations, dim=0)
      
        n = sorted_buf.size(0)
        if n == 0:
            return

        l_idx = int(np.clip(int(n * self.lower_percentile), 0, n - 1))
        u_idx = int(np.clip(int(n * self.upper_percentile), 0, n - 1))

        min_vals = sorted_buf[l_idx]   # shape (d,)
        max_vals = sorted_buf[u_idx]   # shape (d,)
        
        if not hasattr(self, "min") or self.min is None:
            self.min = min_vals.clone()
            self.max = max_vals.clone()
        else:
            self.min = torch.min(self.min, min_vals)
            self.max = torch.max(self.max, max_vals)
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes activation for input x.

        During training:
            - Stores batch activations in curr_task_last_batch.
        
        During evaluation:
            - Stores activations in test_act_buffer for later percentile computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, ...).

        Returns:
            torch.Tensor: Activated tensor of shape (batch, flattened input_shape).
        """
        x_flat = x.view(x.shape[0], -1)
        out = F.leaky_relu(x_flat)

        if self.training:
            self.curr_task_last_batch = out           
        else:
            self.test_act_buffer.extend(list(out.detach().cpu()))

        return out