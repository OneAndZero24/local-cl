from typing import Optional
from copy import deepcopy

import torch
from torch import nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.metric import distillation_loss
from method.method_abc import MethodABC


class LwF(MethodABC):
    """
    Learning without forgetting.
    """

    def __init__(self, 
        module: ActivationRecordingModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        T: float,
        alpha: float,
        gamma: Optional[float]=None,
        clipgrad: Optional[float]=None
    ):
        super().__init__(module, criterion, first_lr, lr, gamma)
        self.task_id = None
        self.T = T
        self.alpha = alpha
        self.clipgrad = clipgrad


    def setup_task(self, task_id: int):
        """
        Task setup.
        """

        self.task_id = task_id
        if task_id > 0:         
            with torch.no_grad():   
                self.old_module = deepcopy(self.module).freeze()
                self.old_module.eval()
        super().setup_optim(task_id)


    def forward(self, x, y):
        """
        Forward pass.
        """
        preds = self.module(x)
        loss = self.add_ael(self.criterion(preds, y))

        if self.task_id > 0:
            with torch.no_grad():
                old_preds = self.old_module(y)

                loss *= self.alpha
                loss += distillation_loss(preds, old_preds, self.T)*(1-self.alpha)
        return loss, preds
    

    def backward(self, loss):
        """
        Backward pass.
        """

        self.optimizer.zero_grad()
        loss.backward()
        if self.clipgrad is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()
