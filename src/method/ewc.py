import logging
from copy import deepcopy

import torch
from torch.nn import functional as F

from method.regularization import param_change_loss
from src.method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class EWC(MethodPluginABC):
    """
    Elastic Weight Consolidation (EWC) method for continual learning.
    EWC is a regularization-based method that mitigates catastrophic forgetting by 
    penalizing changes to important parameters for previously learned tasks.

    Attributes:
        alpha (float): The regularization strength.
        task_id (int or None): The ID of the current task.
        data_buffer (list): A buffer to store data samples.
        params_buffer (dict): A buffer to store model parameters.
        fisher_diag (dict): A dictionary to store the Fisher Information diagonal values.
        head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.

    Methods:
        __init__(alpha: float, head_opt: bool): Initializes the EWC method.
        setup_task(task_id: int): Sets up the task with the given task ID. 
        forward(x, y, loss, preds): Forward.
        _get_fisher_diag(): Compute the diagonal of the Fisher Information Matrix for the model parameters.
    """


    def __init__(self, 
        alpha: float,
        head_opt: bool = True
    ):
        """
        Initializes the EWC (Elastic Weight Consolidation) method.

        Args:
            alpha (float): The regularization strength.
            head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.
        """

        super().__init__()
        self.task_id = None
        self.alpha = alpha
        self.head_opt = head_opt
        log.info(f"Initialized EWC with alpha={alpha}")

        self.data_buffer = set()
        self.params_buffer = {}


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID.
        This method initializes the task by setting the task ID. If the task ID is greater than 0, it 
        freezes the parameters of the module and stores them in the params_buffer. 
        It also calculates the Fisher information diagonal and stores it in 
        fisher_diag. Finally, it initializes an empty data buffer.

        Args:
            task_id (int): The ID of the task to set up.
        """
        

        self.task_id = task_id

        if task_id > 0:
            for name, p in deepcopy(list(self.module.named_parameters())):
                if not self.head_opt and "head" in name:
                    continue
                if p.requires_grad:
                    p.requires_grad = False
                    self.params_buffer[name] = p     
            self.fisher_diag = self._get_fisher_diag()
        self.data_buffer = set()


    def forward(self, x, y, loss, preds):
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

        self.data_buffer.add((x, y))

        if self.task_id > 0:
            loss += self.alpha*param_change_loss(self.module, self.fisher_diag, self.params_buffer, self.head_opt)
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
            if not self.head_opt and "head" in n:
                    continue
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
                if p.requires_grad and (p.grad is not None):
                    if not self.head_opt and "head" in n:
                        continue
                    fisher[n].data += p.grad.data ** 2 / len(self.data_buffer)

        if prev_state:
            self.module.train()
        fisher = {n: p for n, p in fisher.items()}
        return fisher