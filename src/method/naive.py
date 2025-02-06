from typing import Optional

from torch import nn

from model.cl_module_abc import CLModuleABC
from method.method_abc import MethodABC


class Naive(MethodABC):
    """
    Naive joint-training method

    Args:
        module (CLModuleABC): The module to be used for activation recording.
        criterion (nn.Module): The loss function.
        first_lr (float): The initial learning rate.
        lr (float): The learning rate.
        gamma (Optional[float], optional): The learning rate decay factor. Defaults to None.
        reg_type (Optional[str], optional): The type of regularization to be used. Defaults to None.
        clipgrad (Optional[float], optional): The gradient clipping value. Defaults to None.

    Methods:
        setup_task(task_id: int):
            Sets up the task with the given task ID.
        forward(x, y):
            Performs the forward pass and computes the loss and predictions.
        backward(loss):
            Performs the backward pass, applies gradient clipping if specified, and updates the model parameters.
    """
    
    def __init__(self, 
        module: CLModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        gamma: Optional[float]=None,
        reg_type: Optional[str]=None,
        clipgrad: Optional[float]=None
    ):
        """
        Initializes the Naive class.

        Args:
            module (CLModuleABC): The module to be used for activation recording.
            criterion (nn.Module): The loss function to be used.
            first_lr (float): The initial learning rate.
            lr (float): The learning rate.
            gamma (Optional[float], optional): The gamma value for learning rate adjustment. Defaults to None.
            reg_type (Optional[str], optional): The type of regularization to be used. Defaults to None.
            clipgrad (Optional[float], optional): The value to clip gradients. Defaults to None.
        """

        super().__init__(module, criterion, first_lr, lr, reg_type, gamma, clipgrad)


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID.
        This method calls the `setup_optim` method from the superclass to perform
        the necessary setup for the task.

        Args:
            task_id (int): The ID of the task to set up.
        """
        
        super().setup_optim(task_id)
    

    def _forward(self, x, y, loss, preds):
        """
        Perform a forward pass.

        Args:
            x: Input data.
            y: Target data.
            loss: Loss value.
            preds: Predictions.

        Returns:
            Tuple containing the loss and predictions.
        """


        return loss, preds