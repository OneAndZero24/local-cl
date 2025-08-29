import logging

import torch

from src.method.method_plugin_abc import MethodPluginABC
from model.layer.interval_activation import IntervalActivation


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class IntervalPenalization(MethodPluginABC):
    """
    A method plugin that penalizes predictions inside learned interval bounds from IntervalActivation layers.

    Attributes:
        alpha (float): Weight for the penalization term.
        task_id (int or None): Current task identifier.

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


    def setup_task(self, task_id: int):
        """
        Sets the current task identifier.

        Args:
            task_id (int): Identifier for the current task.
        """

        self.task_id = task_id


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

        oobsum = 0
        layers = self.module.layers + [self.module.head]

        if self.task_id > 0:
            for layer in layers:
                if not isinstance(layer, IntervalActivation):
                    continue

                lb, ub = layer.min, layer.max
                # Gather the last `batch_size` activations in one tensor
                acts = torch.stack(layer.curr_task_act_buffer[-x.shape[0]:])  # shape (batch, ...)
                mask = (acts > lb) & (acts < ub)
                penalty = -(acts - lb) * (acts - ub) * mask
                oobsum = oobsum + torch.mean(penalty)

            loss = loss + self.alpha * oobsum
        return loss, preds