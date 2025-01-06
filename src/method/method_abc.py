from typing import Optional
from abc import ABCMeta, abstractmethod

from torch import nn
from torch import optim

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.metric import activation_loss


class MethodABC(metaclass=ABCMeta):
    """
    Abstract base class for all methods.
    """

    def __init__(self, 
        module: ActivationRecordingModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        reg_type: Optional[str]=None,
        gamma: Optional[float]=None
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = None
        self.first_lr = first_lr
        self.lr = lr
        self.reg_type = reg_type
        self.gamma = gamma


    def setup_optim(self, task_id: int):
        """
        Optimizer setup.
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.first_lr if task_id == 0 else self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    def add_ael(self, loss):
        if self.gamma is not None and self.reg_type is not None:
            loss += activation_loss(self.module.activations, self.reg_type, self.gamma)
        return loss


    @abstractmethod
    def setup_task(self, task_id: int):
        """
        Task setup.
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