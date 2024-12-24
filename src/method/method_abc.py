from abc import ABCMeta, abstractmethod

from torch import nn
from torch import optim


class MethodABC(metaclass=ABCMeta):
    """
    Abstract base class for all methods.
    """

    def __init__(self, 
        module: nn.Module,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = None
        self.first_lr = first_lr
        self.lr = lr


    def setup(self, task_id: int):
        """
        Task setup.
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.first_lr if task_id == 0 else self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    @abstractmethod
    def forward(self, x):
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