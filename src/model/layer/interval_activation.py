import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import wandb


class IntervalActivation(nn.Module):
    """
    A neural network module that applies a custom activation function and bounds each element independently based on percentiles.

    Attributes:
        input_shape (tuple or int): Shape of the input tensor (flattened size).
        lower_percentile (float): Lower percentile for min bound calculation.
        upper_percentile (float): Upper percentile for max bound calculation.
        test_act_buffer (list): Stores output samples for percentile calculation.
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
        log_name: str = None,
    ):
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
        
        self.test_act_buffer = [] # samples x input_shape flattened
        self.min = torch.zeros(self.input_shape, requires_grad=True)
        self.max = torch.zeros(self.input_shape, requires_grad=True)
        self.dummy_range = True
        self.log_name = log_name

        self.last_mask = None
        self.curr_task_last_batch = None

    def reset_range(self):
        """
        Compute per-feature percentiles from collected activations and update min/max.
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
        
        self.min = torch.minimum(self.min, min_vals)
        self.max = torch.maximum(self.max, max_vals)
        
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
        Computes activation and saves in buffers.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            out (torch.Tensor): Transformed input.
        """
        x_flat = x.view(x.shape[0], -1)
        out = F.leaky_relu(x_flat)

        if self.training:
            self.curr_task_last_batch = out
            self.last_mask = ((out >= self.min) & (out <= self.max)).float().requires_grad_(True)
        else:
            self.test_act_buffer.extend(list(out.detach().cpu()))

        return out
    
    def register_projection_hooks(self, model):
        if self.last_mask is None or self.curr_task_last_batch is None:
            return

        acts_in_cube = self.curr_task_last_batch * self.last_mask
        a_sum = acts_in_cube.sum()

        J_grads = torch.autograd.grad(
            a_sum,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=True,
            allow_unused=True
        )

        def make_hook(J):
            def hook(grad):
                if grad is None or J is None:
                    return grad
                dot = (grad * J).sum()
                proj = dot / (J.norm()**2 + 1e-8) * J
                return grad - proj
            return hook

        for p, J in zip(model.parameters(), J_grads):
            if p.requires_grad and J is not None:
                p.register_hook(make_hook(J.detach()))
