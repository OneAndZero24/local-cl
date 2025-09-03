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
    new tasks. This is implemented via gradient projection hooks.

    Attributes:
        input_shape (tuple or int): Flattened size of input tensor.
        lower_percentile (float): Lower percentile for min bound computation.
        upper_percentile (float): Upper percentile for max bound computation.
        test_act_buffer (list): Stores activations for percentile computation in eval mode.
        min (torch.Tensor): Lower bound per neuron (updated via reset_range).
        max (torch.Tensor): Upper bound per neuron (updated via reset_range).
        last_mask (torch.Tensor): Binary mask indicating which activations are inside [lb, ub].
        curr_task_last_batch (torch.Tensor): Stores last batch activations during training.
        param_hooks (list): Stores handles to gradient hooks for cube preservation.

    Methods:
        reset_range():
            Computes per-feature min and max bounds using collected activations
            from test_act_buffer. Updates self.min and self.max.
        forward(x):
            Computes Leaky ReLU activation, saves batch activations and mask.
        register_projection_hooks(model):
            Registers hooks on model parameters to project out the gradient components
            that would modify activations inside the [lb, ub] hypercube. Ensures that
            learning outside the cube does not alter stored representations.
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
        self.min = torch.zeros(self.input_shape, requires_grad=True)
        self.max = torch.zeros(self.input_shape, requires_grad=True)
        self.dummy_range = True
        self.log_name = log_name

        self.last_mask = None
        self.curr_task_last_batch = None

        self.param_hooks = []

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
        Computes activation for input x.

        During training:
            - Stores batch activations in curr_task_last_batch.
            - Computes last_mask indicating activations inside the [min, max] hypercube.
        
        During evaluation:
            - Stores activations in test_act_buffer for later percentile computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, ...).

        Returns:
            torch.Tensor: Activated tensor of shape (batch, flattened input_shape).
        """
        x_flat = x.view(x.shape[0], -1)
        out = F.leaky_relu(x_flat)
        device = x_flat.device

        if self.training:
            self.curr_task_last_batch = out
            self.last_mask = ((out >= self.min.to(device)) & (out <= self.max.to(device))).float().requires_grad_(True)
        else:
            self.test_act_buffer.extend(list(out.detach().cpu()))

        return out
    
    def register_projection_hooks(self, model: nn.Module) -> None:
        """
        Registers gradient hooks to preserve activations within [lb, ub].

        Logic:
            1. Computes acts_in_cube = curr_task_last_batch * last_mask.
            2. Computes a scalar sum of acts_in_cube (a_sum).
            3. Computes gradients of a_sum w.r.t. all model parameters (J_grads).
            4. Registers a hook per parameter that projects out the gradient component
               along J_grads, so updates do not change activations inside the cube.
            5. Removes any previously registered hooks before adding new ones.

        Args:
            model (nn.Module): Full model containing this IntervalActivation layer.
        """
        if self.last_mask is None or self.curr_task_last_batch is None:
            return

        # Clear previous hooks
        for handle in self.param_hooks:
            handle.remove()
        self.param_hooks = []

        acts_in_cube = self.curr_task_last_batch * self.last_mask
        a_sum = acts_in_cube.sum()

        J_grads = torch.autograd.grad(
            a_sum,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=True,
            allow_unused=True
        )

        def make_hook(J):
            norm2 = J.norm()**2 + 1e-8
            def hook(grad):
                if grad is None or norm2 < 1e-10:
                    return grad
                dot = (grad * J).sum()
                proj = (dot / norm2) * J
                return grad - proj
            return hook

        for p, J in zip(model.parameters(), J_grads):
            if p.requires_grad and J is not None:
                handle = p.register_hook(make_hook(J.detach()))
                self.param_hooks.append(handle)
