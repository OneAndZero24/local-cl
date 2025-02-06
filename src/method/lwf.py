from typing import Optional
from copy import deepcopy

import torch
from torch import nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from src.method.regularization import distillation_loss
from method.method_abc import MethodABC


class LwF(MethodABC):
    """
    LwF (Learning without Forgetting) method for continual learning.

    Attributes:
        module (ActivationRecordingModuleABC): The module to be trained.
        criterion (nn.Module): The loss function.
        first_lr (float): The initial learning rate.
        lr (float): The learning rate.
        T (float): The temperature for distillation loss.
        alpha (float): The weight for the distillation loss.
        gamma (Optional[float]): The regularization parameter.
        reg_type (Optional[str]): The type of regularization.
        clipgrad (Optional[float]): The gradient clipping value.
        task_id (int): The current task identifier.
        old_module (ActivationRecordingModuleABC): The module from the previous task.

    Methods:
        __init__(module, criterion, first_lr, lr, T, alpha, gamma=None, reg_type=None, clipgrad=None):
            Initializes the LwF method with the given parameters.
        setup_task(task_id):
            Sets up the task by initializing the task identifier and preparing the old module for distillation.
        forward(x, y):
            Performs the forward pass, computes the loss, and applies the distillation loss if not the first task.
        backward(loss):
            Performs the backward pass, applies gradient clipping if specified, and updates the model parameters.
    """

    def __init__(self, 
        module: ActivationRecordingModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        T: float,
        alpha: float,
        gamma: Optional[float]=None,
        reg_type: Optional[str]=None,
        clipgrad: Optional[float]=None
    ):
        """
        Initialize the LwF (Learning without Forgetting) method.

        Args:
            module (ActivationRecordingModuleABC): The module to record activations.
            criterion (nn.Module): The loss function.
            first_lr (float): The initial learning rate.
            lr (float): The learning rate.
            T (float): The temperature for knowledge distillation.
            alpha (float): The weight for the distillation loss.
            gamma (Optional[float]): The regularization parameter. Default is None.
            reg_type (Optional[str]): The type of regularization. Default is None.
            clipgrad (Optional[float]): The gradient clipping value. Default is None.
        """
                
        super().__init__(module, criterion, first_lr, lr, reg_type, gamma)
        self.task_id = None
        self.T = T
        self.alpha = alpha
        self.clipgrad = clipgrad


    def setup_task(self, task_id: int):
        """
        Sets up the task for the given task ID.
        This method initializes the task by setting the task ID and calling the 
        setup_optim method from the superclass. If the task ID is greater than 0, 
        it creates a deep copy of the current module, disables gradient computation 
        for the copied module's parameters, and sets the copied module to evaluation mode.

        Args:
            task_id (int): The ID of the task to set up.
        """

        self.task_id = task_id
        super().setup_optim(task_id)
        if task_id > 0:         
            with torch.no_grad():   
                self.old_module = deepcopy(self.module)
                for p in self.old_module.parameters():
                    p.requires_grad = False
                self.old_module.eval()


    def forward(self, x, y):
        """
        Perform a forward pass through the network and compute the loss.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            tuple: A tuple containing the computed loss and the predictions.
        """
        
        preds = self.module(x)
        loss = self.add_ael(self.criterion(preds, y))

        if self.task_id > 0:
            with torch.no_grad():
                old_preds = self.old_module(x)

            loss *= self.alpha
            loss += distillation_loss(preds, old_preds, self.T)*(1-self.alpha)
        return loss, preds
    

    def backward(self, loss):
        """
        Perform a backward pass and update the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
            
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
