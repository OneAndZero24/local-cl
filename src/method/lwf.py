import logging
from copy import deepcopy

import torch

from method.regularization import distillation_loss
from src.method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class LwF(MethodPluginABC):
    """
    LwF (Learning without Forgetting) method for continual learning.
    This class implements the Learning without Forgetting (LwF) method, which helps in retaining 
    knowledge from previous tasks while learning new tasks. It uses a distillation loss to 
    preserve the performance on old tasks.

    Attributes:
        T (float): Temperature scaling parameter for distillation.
        alpha (float): Weighting factor for the distillation loss.

    Methods:
        __init__(T: float, alpha: float):
            Initializes the LwF method with the given temperature and alpha values.
        setup_task(task_id: int):
            Sets up the task for the given task ID. If the task ID is greater than 0, 
        forward(x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, preds: torch.Tensor) -> tuple:
            Performs a forward pass and computes the loss with optional distillation. 
            If the task ID is greater than 0, it computes the distillation loss using the 
            predictions from the old module and combines it with the initial loss.
    """

    def __init__(self, 
        T: float,
        alpha: float,
    ):
        """
        Initializes the LwF (Learning without Forgetting) method.

        Args:
            T (float): Temperature scaling parameter for distillation.
            alpha (float): Weighting factor for the distillation loss.
        """

        super().__init__()
        self.task_id = None
        self.T = T
        self.alpha = alpha
        log.info(f"Initialized LwF with T={T}, alpha={alpha}")


    def setup_task(self, task_id: int):
        """
        Sets up the task for the given task ID.
        This method initializes the task by setting the task ID. If the task ID is greater than 0, 
        it creates a deep copy of the current module, disables gradient computation 
        for the copied module's parameters, and sets the copied module to evaluation mode.

        Args:
            task_id (int): The ID of the task to set up.
        """

        self.task_id = task_id
        if task_id > 0:         
            with torch.no_grad():
                for module in self.module.modules():
                    if type(module).__name__ == "IntervalActivation":
                        del module.curr_task_last_batch
                self.old_module = deepcopy(self.module)
                for p in self.old_module.parameters():
                    p.requires_grad = False
                self.old_module.eval()


    def forward(self, x, y, loss, preds):
        """
        Perform a forward pass and compute the loss with optional distillation.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Initial loss value.
            preds (torch.Tensor): Predictions from the current model.
            
        Returns:
            tuple: A tuple containing the updated loss and predictions.
        """

        if self.task_id > 0:
            with torch.no_grad():
                old_preds = self.old_module(x)

            loss += self.alpha*distillation_loss(preds, old_preds, self.T)
        return loss, preds