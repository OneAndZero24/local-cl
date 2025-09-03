import logging
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class IntervalPenalization(MethodPluginABC):
    """
    IntervalPenalization plugin for continual learning with IntervalActivation layers.

    This plugin enforces two objectives:
      1. **Variance minimization**: Encourages activations in IntervalActivation layers
         to have low variance, which stabilizes the representation.
      2. **Output preservation**: Preserves activations and parameter outputs
         of IntervalActivation layers learned from previous tasks, ensuring minimal
         forgetting when learning new tasks.

    Attributes:
        var_scale (float): Weight for the variance regularization term.
        output_reg_scale (float): Weight for the output preservation term.
        task_id (int or None): Identifier of the current task.
        params_buffer (dict): Stores frozen parameter values from the previous task.
        input_shape (tuple or None): Shape of flattened inputs (for bookkeeping).
        old_state (dict): Snapshot of parameters and buffers from the previous task.

    Methods:
        setup_task(task_id):
            Configures the plugin for the current task, freezes parameters from
            previous tasks, and snapshots the old state.
        forward_with_snapshot(x, stop_at="IntervalActivation"):
            Performs a forward pass using frozen previous-task parameters,
            stopping at the first IntervalActivation layer.
        snapshot_state():
            Returns a dictionary with clones of current parameters and buffers.
        forward(x, y, loss, preds):
            Adds variance and interval-based regularization to the input loss.
    """

    def __init__(self,
            var_scale: float = 0.01,
            output_reg_scale: float = 1.0,
        ) -> None:
        """
        Initializes the IntervalPenalization plugin.

        Args:
            var_scale (optional, float): Weight for output preservation.
            output_reg_scale (optional, float): Weight for output preservation.

        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with var_scale={var_scale} and output_reg_scale={output_reg_scale}")

        self.var_scale = var_scale
        self.output_reg_scale = output_reg_scale

        self.input_shape = None
        self.params_buffer = {}

    def forward_with_snapshot(self, x: torch.Tensor, stop_at: str="IntervalActivation") -> torch.Tensor:
        """
        Runs a forward pass using frozen parameters and buffers from the previous task.

        The forward stops at the first IntervalActivation layer (by default) to
        capture intermediate representations for interval preservation.

        Args:
            x (torch.Tensor): Input tensor.
            stop_at (str, optional): Name of the layer class at which to stop.

        Returns:
            torch.Tensor: Output of the forward pass up to the stop layer.
        """
        saved_param_datas = {name: param.data for name, param in self.module.named_parameters()}
        saved_buffers = {name: buf for name, buf in self.module.named_buffers()}

        for name, param in self.module.named_parameters():
            param.data = self.old_state["params"][name].clone()
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = self.old_state["buffers"][name].clone()

        out = x
        for layer in self.module.layers:
            out = layer(out)
            if type(layer).__name__ == stop_at:
                break

        for name, param in self.module.named_parameters():
            param.data = saved_param_datas[name]
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = saved_buffers[name]

        return out

    @torch.no_grad()
    def snapshot_state(self) -> dict:
        """
        Captures a snapshot of the model's current parameters and buffers.

        Returns:
            dict: {
                "params": OrderedDict of parameter clones,
                "buffers": OrderedDict of buffer clones
            }
        """
        return {
            "params": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_parameters()),
            "buffers": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_buffers()),
        }


    def setup_task(self, task_id: int) -> None:
        """
        Sets the current task identifier.

        Args:
            task_id (int): Identifier for the current task.
        """

        self.task_id = task_id
        if task_id > 0:
            self.params_buffer = {}
            for name, p in deepcopy(list(self.module.named_parameters())):
                if p.requires_grad:
                    p.requires_grad = False
                    self.params_buffer[name] = p.detach().clone()
            self.old_state = self.snapshot_state()
                    
    def forward(self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, 
                preds: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Applies interval regularization and variance penalties to the loss.

        For each IntervalActivation layer:
            1. Computes the batch variance of activations and accumulates it in var_loss.
            2. If learning a new task, computes output regularization to preserve
               activations inside the [min, max] interval from the previous task.
            3. Adds variance and output regularization terms to the original loss.

        Args:
            x (torch.Tensor): Input batch.
            y (torch.Tensor): Target batch.
            loss (torch.Tensor): Original loss value.
            preds (torch.Tensor): Model predictions (optional for this method).

        Returns:
            tuple: (loss_with_penalty, preds)
              - loss_with_penalty (torch.Tensor): Original loss plus interval and variance penalties.
              - preds (torch.Tensor): Unmodified predictions.
        """

        x = x.flatten(start_dim=1)
        self.input_shape = x.shape

        layers = self.module.layers + [self.module.head]

        var_loss = 0.0
        output_reg_loss = 0.0

        for idx, layer in enumerate(layers):
            if not type(layer).__name__ == "IntervalActivation":
                continue
            
            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:

                lb = layer.min
                ub = layer.max
            
                # Regularization of learnable parameters above the IntervalActivation layer
                next_layer = layers[idx + 1]
                if hasattr(next_layer, "classifier"):
                    lower_bound_reg = 0.0
                    upper_bound_reg = 0.0
                    for name, p in next_layer.classifier.named_parameters():
                        for mod_name, mod_param in self.module.named_parameters():
                            if mod_param is p and mod_name in self.params_buffer:
                                prev_param = self.params_buffer[mod_name]
                                if "weight" in name:
                                    weight_diff = p - prev_param

                                    weight_diff_pos = torch.relu(weight_diff)
                                    weight_diff_neg = torch.relu(-weight_diff)

                                    lower_bound_reg += weight_diff_pos @ lb - weight_diff_neg @ ub
                                    upper_bound_reg += weight_diff_pos @ ub - weight_diff_neg @ lb

                                elif "bias" in name:
                                    lower_bound_reg += p - prev_param
                                    upper_bound_reg += p - prev_param

                output_reg_loss += lower_bound_reg.sum().pow(2) + upper_bound_reg.sum().pow(2)
    
        loss = loss + self.var_scale * var_loss \
                + self.output_reg_scale * output_reg_loss
        return loss, preds