import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from typing import List


class IntervalActivation(nn.Module):
    """
    IntervalActivation layer for preserving learned representations within a hypercube.

    This layer applies a Leaky ReLU activation and tracks the range of activations 
    across batches. It defines a [lb, ub] hypercube per neuron, which can be used 
    to enforce that activations within this cube remain unchanged when learning 
    new tasks.

    Attributes:
        layer_name (str): Optional name of the layer for logging purposes.
        lower_percentile (float): Lower percentile for min bound computation.
        upper_percentile (float): Upper percentile for max bound computation.
        min (torch.Tensor): Lower bound per neuron (updated via reset_range).
        max (torch.Tensor): Upper bound per neuron (updated via reset_range).
        curr_task_last_batch (torch.Tensor): Stores last batch activations during training.
        use_nonlinear_transform (bool): If True, nonlinear transform is used.

    Methods:
        reset_range(acttivations):
            Computes per-feature min and max bounds using collected `activations`.
            Updates self.min and self.max.
        forward(x):
            Computes Leaky ReLU activation.
    """

    def __init__(self,
        layer_name: str = None,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        log_name: str = None,
        use_nonlinear_transform: bool = True
    ) -> None:
        """
        Initializes the IntervalActivation layer.

        Args:
            layer_name (str, optional): Name of the layer. Defaults to None.
            lower_percentile (float, optional): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float, optional): Upper percentile for max bound. Defaults to 0.95.
            log_name (str, optional): Name of the layer for wandb logging. Defaults to None.
            use_nonlinear_transform (bool): If True, nonlinear transform is used.
        """

        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        self.min = None
        self.max = None
        self.log_name = log_name
        self.layer_name = layer_name

        self.curr_task_last_batch = None
        self.use_nonlinear_transform = use_nonlinear_transform

    def reset_range(self, activations: List[torch.Tensor]) -> None:
        """
        Update the per-neuron activation interval ([min, max]) using collected activations.

        This method computes a robust estimate of the activation range for each neuron 
        based on a list of activation tensors. It updates `self.min` and `self.max` 
        either by initializing them (if not already set) or by taking the element-wise 
        min/max with existing values. Optionally, it logs per-neuron intervals to wandb.

        Steps:
            1. Stack the list of activation tensors into a single [n_samples, d] tensor.
            2. Sort activations along the sample dimension and select lower/upper percentiles.
            3. Update `self.min` and `self.max` by element-wise min/max.
            4. Optionally log per-neuron min, max, and interval size to wandb.

        Args:
            activations (List[torch.Tensor]): A list of activation tensors for a batch of inputs.
                Each tensor should have shape [batch_size, num_neurons].

        Side Effects:
            - Updates `self.min` and `self.max`.
            - Logs interval statistics to wandb if `self.log_name` is set and a wandb run exists.
        """
        
        if len(activations) == 0:
            return

        activations = torch.cat(activations, dim=0).to(activations[0].device)  # shape: [n_samples, d]
        sorted_buf, _ = torch.sort(activations, dim=0)
      
        n = sorted_buf.size(0)
        if n == 0:
            return

        l_idx = int(np.clip(int(n * self.lower_percentile), 0, n - 1))
        u_idx = int(np.clip(int(n * self.upper_percentile), 0, n - 1))

        min_vals = sorted_buf[l_idx]   # shape (d,)
        max_vals = sorted_buf[u_idx]   # shape (d,)
        
        if self.min is None or self.max is None:
            self.min = min_vals.clone()
            self.max = max_vals.clone()
        else:
            self.min = torch.minimum(self.min, min_vals)
            self.max = torch.maximum(self.max, max_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes activation for input x.

        During training:
            - Stores batch activations in curr_task_last_batch.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, ...).

        Returns:
            torch.Tensor: Activated tensor.
        """
        if self.use_nonlinear_transform:
            x = F.leaky_relu(x)

        if self.training:
            self.curr_task_last_batch = x

        return x
    
    def __repr__(self):
        return super().__repr__() + f"(layer_name={self.layer_name})"