import torch
from torch import nn
import numpy as np
import wandb


class IntervalLayer(nn.Linear):
    """
    Linear layer with interval tracking.
    - Behaves like a normal nn.Linear in forward.
    - Tracks activation ranges for interval preservation.
    - Constraint handling is done during backward.
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        log_name: str = None,
    ):
        """
        Args:
            input_shape (int): Flattened input size (in_features).
            output_shape (int): Output size (out_features).
            lower_percentile (float): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float): Upper percentile for max bound. Defaults to 0.95.
            log_name (str, optional): Name for wandb logging. Defaults to None.
        """
        super().__init__(in_features=input_shape, out_features=output_shape, bias=True)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        # Buffers for interval calculation
        self.curr_task_last_batch = None
        self.test_act_buffer = []  
        self.register_buffer("min", torch.zeros(self.input_shape))
        self.register_buffer("max", torch.zeros(self.input_shape))
        self.log_name = log_name

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard linear transformation + buffer collection.
        Interval masking is not done here (constraint handled in backward).
        """
        x_flat = x.view(x.shape[0], -1)
        out = super().forward(x_flat)

        if not self.training:
            self.test_act_buffer.extend(list(x_flat.detach().cpu()))

        if self.training:
            self.curr_task_last_batch = x_flat

        return out
