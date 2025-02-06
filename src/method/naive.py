from typing import Optional

import torch
from torch import nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.method_abc import MethodABC


class Naive(MethodABC):
    """
    Naive joint-training method

    Args:
        module (ActivationRecordingModuleABC): The module to be used for activation recording.
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
        module: ActivationRecordingModuleABC,
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
            module (ActivationRecordingModuleABC): The module to be used for activation recording.
            criterion (nn.Module): The loss function to be used.
            first_lr (float): The initial learning rate.
            lr (float): The learning rate.
            gamma (Optional[float], optional): The gamma value for learning rate adjustment. Defaults to None.
            reg_type (Optional[str], optional): The type of regularization to be used. Defaults to None.
            clipgrad (Optional[float], optional): The value to clip gradients. Defaults to None.
        """

        super().__init__(module, criterion, first_lr, lr, reg_type, gamma)
        self.clipgrad = clipgrad


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID.
        This method calls the `setup_optim` method from the superclass to perform
        the necessary setup for the task.

        Args:
            task_id (int): The ID of the task to set up.
        """
        
        super().setup_optim(task_id)
    

    def forward(self, x, y):
        """
        Perform a forward pass through the model.

        Args:
            x: Input data.
            y: Target data.
            
        Returns:
            A tuple containing:
                - The result of applying the criterion to the predictions and targets, 
                  with additional processing by add_ael.
                - The raw predictions from the model.
        """

        preds = self.module(x)
        return self.add_ael(self.criterion(preds, y)), preds
    

    def backward(self, loss):
        """
        Perform a backward pass and update model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which gradients will be computed.
            
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Optionally clips the gradients to prevent exploding gradients.
        4. Updates the model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        loss.backward()
        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()