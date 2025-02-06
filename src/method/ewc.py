from typing import Optional
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from model.activation_recording_abc import ActivationRecordingModuleABC
from src.method.regularization import ewc_loss
from method.method_abc import MethodABC


class EWC(MethodABC):
    """
    Elastic Weight Consolidation (EWC) method for continual learning.
    EWC is a regularization-based method that mitigates catastrophic forgetting by 
    penalizing changes to important parameters for previously learned tasks.

    Attributes:
        task_id (int): Identifier for the current task.
        clipgrad (float): Gradient clipping value.
        alpha (float): Weighting factor for the EWC loss.
        data_buffer (list): Buffer to store data samples.
        params_buffer (dict): Buffer to store parameters of the model.
        fisher_diag (dict): Diagonal of the Fisher Information Matrix.

    Methods:
        __init__(module, criterion, first_lr, lr, alpha, gamma=None, reg_type=None, clipgrad=None):
            Initializes the EWC method with the given parameters.
        setup_task(task_id):
            Sets up the task with the given task identifier.
        forward(x, y):
            Performs the forward pass and computes the loss.
        backward(loss):
            Performs the backward pass and updates the model parameters.
        _get_fisher_diag():
            Calculates the diagonal of the Fisher Information Matrix.
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
        """
        Initializes the EWC (Elastic Weight Consolidation) method.

        Args:
            module (ActivationRecordingModuleABC): The module to be used for activation recording.
            criterion (nn.Module): The loss function to be used.
            first_lr (float): The initial learning rate.
            lr (float): The learning rate for subsequent updates.
            alpha (float): The regularization strength.
            gamma (Optional[float], optional): The scaling factor for the regularization term. Defaults to None.
            reg_type (Optional[str], optional): The type of regularization to be used. Defaults to None.
            clipgrad (Optional[float], optional): The gradient clipping value. Defaults to None.
        """

        super().__init__(module, criterion, first_lr, lr, reg_type, gamma, clipgrad)
        self.task_id = None
        self.alpha = alpha

        self.data_buffer = []
        self.params_buffer = {}


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID.
        This method initializes the task by setting the task ID and calling the 
        parent class's setup_optim method. If the task ID is greater than 0, it 
        freezes the parameters of the module and stores them in the params_buffer. 
        It also calculates the Fisher information diagonal and stores it in 
        fisher_diag. Finally, it initializes an empty data buffer.

        Args:
            task_id (int): The ID of the task to set up.
        """
        

        self.task_id = task_id
        super().setup_optim(task_id)

        if task_id > 0:
            for name, p in deepcopy(list(self.module.named_parameters())):
                p.requires_grad = False
                self.params_buffer[name] = p     
            self.fisher_diag = self._get_fisher_diag()
        self.data_buffer = []


    def _forward(self, x, y, loss, preds):
        """
        Perform a forward pass and compute the loss with Elastic Weight Consolidation (EWC) regularization.

        Args:
            x (Tensor): Input data.
            y (Tensor): Target labels.
            loss (Tensor): Computed loss from the model's predictions.
            preds (Tensor): Model's predictions.
            
        Returns:
            Tuple[Tensor, Tensor]: The adjusted loss with EWC regularization and the model's predictions.
        """

        self.data_buffer.append((x, y))

        if self.task_id > 0:
            loss *= self.alpha
            loss += (1-self.alpha)*ewc_loss(self.module, self.fisher_diag, self.params_buffer)
        return loss, preds


    def _get_fisher_diag(self):
        """
        Compute the diagonal of the Fisher Information Matrix for the model parameters.
        This method calculates the Fisher Information Matrix (FIM) diagonal for the parameters
        of the model. The FIM is used in various machine learning algorithms, particularly in
        continual learning, to measure the importance of each parameter.
        
        Returns:
            dict: A dictionary where keys are parameter names and values are the corresponding
                  Fisher Information values.
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