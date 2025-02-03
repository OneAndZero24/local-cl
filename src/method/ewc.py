from typing import Optional
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from model.activation_recording_abc import ActivationRecordingModuleABC
from method.metric import ewc_loss
from method.method_abc import MethodABC


class EWC(MethodABC):
    """
    Elastic Weight Consolidation.
    """

    def __init__(self, 
        module: ActivationRecordingModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        alpha: float,
        gamma: Optional[float]=None,
        reg_type: Optional[str]=None,
        clipgrad: Optional[float]=None
    ):
        super().__init__(module, criterion, first_lr, lr, reg_type, gamma)
        self.task_id = None
        self.clipgrad = clipgrad
        self.alpha = alpha

        self.data_buffer = []
        self.params_buffer = {}


    def setup_task(self, task_id: int):
        """
        Task setup.
        """

        self.task_id = task_id
        super().setup_optim(task_id)

        if task_id > 0:
            for name, p in deepcopy(list(self.module.named_parameters())):
                p.requires_grad = False
                self.params_buffer[name] = p     
            self.fisher_diag = self._get_fisher_diag()
        self.data_buffer = []


    def forward(self, x, y):
        """
        Forward pass.
        """

        self.data_buffer.append((x, y))
        preds = self.module(x)
        loss = self.add_ael(self.criterion(preds, y))

        if self.task_id > 0:
            loss *= self.alpha
            loss += (1-self.alpha)*ewc_loss(self.module, self.fisher_diag, self.params_buffer)
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


    def _get_fisher_diag(self):
        """
        Calculates fisher matrix diagonal
        """

        params = {name: p for name, p in self.module.named_parameters() if p.requires_grad}
        fisher = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            fisher[n] = torch.autograd.Variable(p.data)

        prev_state = self.module.training
        self.module.eval()
        for x, y in self.data_buffer:
            self.module.zero_grad()
            output = self.module(x)
            label = y
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            negloglikelihood.backward()

            for n, p in self.module.named_parameters():
                fisher[n].data += p.grad.data ** 2 / len(self.data_buffer)

        if prev_state:
            self.module.train()
        fisher = {n: p for n, p in fisher.items()}
        return fisher