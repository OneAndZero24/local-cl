import logging

from src.method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class IntervalPenalization(MethodPluginABC):
    """
    A method plugin that minimizes variance across predictions from IntervalLayer layers.

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

        layers = self.module.layers + [self.module.head]

        if self.task_id > 0:
            for layer in layers:
                if not type(layer).__name__ == "IntervalLayer":
                    continue

                acts = layer.curr_task_last_batch
                acts_flat = acts.view(acts.size(0), -1)
                batch_var = acts_flat.var(dim=0, unbiased=False).mean()          

            loss = loss + self.alpha * batch_var
        return loss, preds