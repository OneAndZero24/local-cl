from typing import Optional

from torch import nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.method_abc import MethodABC


class Naive(MethodABC):
    """
    Naive joint training method.
    """

    def __init__(self, 
        module: ActivationRecordingModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        gamma: Optional[float]=None
    ):
        super().__init__(module, criterion, first_lr, lr, gamma)


    def setup_task(self, task_id: int):
        """
        Task setup.
        """

        pass
    

    def forward(self, x, y):
        """
        Forward pass.
        """
        preds = self.module(x)
        return self.add_ael(self.criterion(preds, y)), preds
    

    def backward(self, loss):
        """
        Backward pass.
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()