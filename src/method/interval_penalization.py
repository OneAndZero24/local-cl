import logging
from copy import deepcopy

import torch

from src.method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class IntervalPenalization(MethodPluginABC):
    """
    A method plugin that minimizes variance across predictions from IntervalActivation layers.

    Attributes:
        alpha (float): Weight for the penalization term.
        task_id (int or None): Current task identifier.
        params_buffer (dict): Optimal parameters from the last task.

    Methods:
        setup_task(task_id):
            Sets the current task identifier.
        forward(x, y, loss, preds):
            Adds penalization to the loss for predictions inside interval bounds.
    """

    def __init__(self,
            alpha: float = 0.01,
        ):
        """
        Initializes the IntervalPenalization plugin.

        Args:
            alpha (float, optional): Weight for the penalization term. Defaults to 0.01.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with alpha={alpha}")

        self.alpha = alpha
        self.params_buffer = {}


    def setup_task(self, task_id: int):
        """
        Sets the current task identifier.

        Args:
            task_id (int): Identifier for the current task.
        """

        self.task_id = task_id

        if task_id > 0:
            for name, p in deepcopy(list(self.module.named_parameters())):
                if p.requires_grad:
                    p.requires_grad = False
                    self.params_buffer[name] = p     


    def forward(self, x, y, loss, preds):
        """
        Adds penalization to the loss for predictions outside interval bounds.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Current loss value.
            preds (torch.Tensor): Model predictions.

        Returns:
            tuple: (loss, preds) with penalization added to loss.
        """

        layers = self.module.layers + [self.module.head]
        var_loss = 0.0
        output_reg_loss = 0.0

        for idx, layer in enumerate(layers):
            if not type(layer).__name__ == "IntervalActivation":
                continue
            
            # Calculate variance
            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                # Regularize head outpus
                lb = layer.min
                ub = layer.max

                # Regularization of the *next* layer (weights on top of IntervalActivation)
                next_layer = layers[idx + 1]
                if hasattr(next_layer, "classifier"):
                    lower_bound_reg = 0.0
                    upper_bound_reg = 0.0
                    for name, p in next_layer.classifier.named_parameters():
                        for mod_name, mod_param in self.module.named_parameters():
                            if mod_param is p and mod_name in self.params_buffer:
                                prev_param = self.params_buffer[mod_name].to(p.device)
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
    
        loss = loss + self.alpha * (batch_var + output_reg_loss)
        return loss, preds