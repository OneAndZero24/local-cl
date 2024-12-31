from typing import Optional

import torch
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
        gamma: Optional[float]=None,
        clipgrad: Optional[float]=None
    ):
        super().__init__(module, criterion, first_lr, lr, gamma)
        self.clipgrad = clipgrad


    def setup_task(self, task_id: int):
        """
        Task setup.
        """

        super().setup_optim(task_id)
    

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
        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()