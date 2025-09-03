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
    A method plugin that minimizes variance across predictions from IntervalActivation layers, and
    preserves outputs from those layers.

    Attributes:
        var_scale (float): Weight for the variance term.
        output_reg_scale (float): Weight for output preservation.
        task_id (int or None): Current task identifier.
        params_buffer (dict): Optimal parameters from the last task.

    Methods:
        setup_task(task_id):
            Sets the current task identifier.
        forward(x, y, loss, preds):
            Adds penalization to the loss for predictions inside interval bounds.
    """

    def __init__(self,
            var_scale: float = 0.01,
            output_reg_scale: float = 1.0,
        ):
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

    def forward_with_snapshot(self, x, stop_at="IntervalActivation"):
        """
        Run the model forward using frozen params/buffers, stopping at the first IntervalActivation.
        """
        # Save references to current parameter data and buffer tensors
        saved_param_datas = {name: param.data for name, param in self.module.named_parameters()}
        saved_buffers = {name: buf for name, buf in self.module.named_buffers()}

        # Set parameters to snapshot values (using clones to avoid inplace on originals)
        for name, param in self.module.named_parameters():
            param.data = self.old_state["params"][name].clone()
        
        # Set buffers to snapshot values (using clones)
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = self.old_state["buffers"][name].clone()

        # Run the forward pass
        out = x
        for layer in self.module.layers:
            out = layer(out)
            if type(layer).__name__ == stop_at:
                break

        # Restore original parameter data and buffer tensors
        for name, param in self.module.named_parameters():
            param.data = saved_param_datas[name]
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = saved_buffers[name]

        return out

    @torch.no_grad()
    def snapshot_state(self):
        return {
            "params": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_parameters()),
            "buffers": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_buffers()),
        }


    def setup_task(self, task_id: int):
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
                
            # To debug: If those layers are frozen, BUT
            # a classification head is unfrozen, then we have
            # zero forgetting!
            for layer in self.module.layers:
                if type(layer).__name__ == "Linear":
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                    
    def forward(self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, 
                preds: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Adds interval regularization.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Current loss value.
            preds (torch.Tensor): Model predictions.

        Returns:
            tuple: (loss, preds) with penalization added to loss.
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