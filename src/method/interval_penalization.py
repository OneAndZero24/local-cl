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
        batch_size = x.shape[0]
        for layer in self.module.layers+[self.module.head, self.module.neck if hasattr(self, 'neck') else None]:
            if isinstance(layer, IntervalActivation):
                lower_bound = layer.min
                upper_bound = layer.max
                for i in range(batch_size):
                    activation = layer.buffer[-(i+1)]  # Get from last to last-batch_size
                    oobsum += torch.sum(
                        torch.where(
                            (activation < upper_bound) & (activation > lower_bound),
                            (-1) * (activation - lower_bound) * (activation - upper_bound),
                            torch.zeros_like(activation)
                        )
                    )

        loss += self.alpha*oobsum
        return loss, preds