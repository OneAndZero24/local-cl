from torch import nn

from method_abc import MethodABC


class Naive(MethodABC):
    """
    Naive method.
    """

    def __init__(self, 
        module: nn.Module,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float
    ):
        super().__init__(module, criterion, first_lr, lr)


    def forward(self, x, y):
        """
        Forward pass.
        """
        preds = self.module(x)
        return self.criterion(preds, y)
    

    def backward(self, loss):
        """
        Backward pass.
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()