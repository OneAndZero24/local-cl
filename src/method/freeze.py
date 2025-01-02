from typing import Optional

import torch
from torch import nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.method_abc import MethodABC

from copy import deepcopy


class Freeze(MethodABC):
    """
    Freeze task-specific parameters for each task.
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

        self.init_weight_mask()

    def init_weight_mask(self):
        self.weight_mask = deepcopy(self.module.state_dict())
        for key in self.weight_mask:
            self.weight_mask[key] = torch.ones_like(self.weight_mask[key])

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

        for name, param in self.module.named_parameters():
            if param.grad is not None and name in self.weight_mask:
                param.grad.data.mul_(self.weight_mask[name])

        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()