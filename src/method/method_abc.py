from typing import Optional
from abc import ABCMeta, abstractmethod

from torch import nn
from torch import optim

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.regularization import regularization


class MethodABC(metaclass=ABCMeta):
    """
    An abstract base class for methods using activation recording modules.

    Attributes:
        module (ActivationRecordingModuleABC): The activation recording module.
        criterion (nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer, initialized as None.
        first_lr (float): The initial learning rate for the first task.
        lr (float): The learning rate for subsequent tasks.
        reg_type (Optional[str]): The type of regularization, if any.
        gamma (Optional[float]): The regularization parameter, if any.

    Methods:
        setup_optim(task_id: int):
            Sets up the optimizer for the given task.
        add_reg(loss):
            Adds activation entropy loss (AEL) to the given loss if regularization is specified.
        setup_task(task_id: int):
            Abstract method for setting up a task. Must be implemented by subclasses.
        forward(x, y):
            Abstract method for the forward pass. Must be implemented by subclasses.
        backward(loss):
            Abstract method for the backward pass. Must be implemented by subclasses.
    """

    def __init__(self, 
        module: ActivationRecordingModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        reg_type: Optional[str]=None,
        gamma: Optional[float]=None,
        clipgrad: Optional[float]=None
    ):
        """
        Initializes the MethodABC class with the given parameters.

        Args:
            reg_type (Optional[str], optional): The type of regularization, if any. Defaults to None.
            gamma (Optional[float], optional): The regularization parameter, if any. Defaults to None.
        """

        self.module = module
        self.criterion = criterion
        self.optimizer = None
        self.first_lr = first_lr
        self.lr = lr
        self.reg_type = reg_type
        self.gamma = gamma
        self.clipgrad = clipgrad


    def setup_optim(self, task_id: int):
        """
        Sets up the optimizer for the model.
        This method initializes the optimizer for the model's parameters. It filters out 
        parameters that do not require gradients and sets the learning rate based on the 
        task ID. If the task ID is 0, it uses the initial learning rate (`first_lr`), 
        otherwise it uses the standard learning rate (`lr`).

        Args:
            task_id (int): The ID of the current task. Determines which learning rate to use.

        Returns:
            None
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.first_lr if task_id == 0 else self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    def add_reg(self, loss):
        """
        Adjusts the given loss by adding regularization.

        Args:
            loss (float): The original loss value to be adjusted.

        Returns:
            float: The adjusted loss value.
        """

        if self.gamma is not None and self.reg_type is not None:
            loss = (1-self.gamma)*loss+self.gamma*regularization(self.module.activations, self.reg_type)
        return loss


    @abstractmethod
    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID.
        
        Args:
            task_id (int): The unique identifier of the task to be set up.
        """

        pass


    @abstractmethod
    def _forward(self, x, y, loss, preds):
        """
        Internal forward pass.
        """

        pass


    def forward(self, x, y):
        """
        Perform a forward pass, compute the loss, and return predictions.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            tuple: A tuple containing the computed loss and the predictions.
        """

        preds = self.module(x)
        loss = self.criterion(preds, y)
        loss = self.add_reg(loss)

        return self._forward(x, y, loss, preds)

    def backward(self, loss):
        """
        Perform a backward pass and update the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Optionally clips the gradients to a maximum norm if `self.clipgrad` is set.
        4. Updates the model parameters using the optimizer.
        """     

        self.optimizer.zero_grad()
        loss.backward()
        if self.clipgrad is not None:
            nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()