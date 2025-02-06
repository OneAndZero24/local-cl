from typing import Optional
from abc import ABCMeta, abstractmethod

from torch import nn
from torch import optim

from model.activation_recording_abc import ActivationRecordingModuleABC
from src.method.regularization import activation_loss


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
        add_ael(loss):
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
        gamma: Optional[float]=None
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


    def add_ael(self, loss):
        """
        Adjusts the given loss by adding an activation loss element (AEL) if certain conditions are met.

        Args:
            loss (float): The original loss value to be adjusted.

        Returns:
            float: The adjusted loss value.
        """

        if self.gamma is not None and self.reg_type is not None:
            loss = (1-self.gamma)*loss+activation_loss(self.module.activations, self.reg_type, self.gamma)
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
    def forward(self, x, y):
        """
        Forward pass.
        """

        pass


    @abstractmethod
    def backward(self, loss):
        """
        Backward pass.
        """

        pass