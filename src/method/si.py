import logging

import torch

from method.regularization import param_change_loss
from src.method.method_plugin_abc import MethodPluginABC
from util import pad_zero_dim0


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class SI(MethodPluginABC):
    """
    SI (Synaptic Intelligence) is a method plugin for continual learning that helps mitigate catastrophic forgetting by 
    regularizing the changes in the parameters of the neural network.

    Attributes:
        alpha (float): The regularization strength.
        eps (float): A small value to avoid division by zero.
        task_id (int): The current task identifier.
        prev_param (dict): A dictionary to store the previous parameters of the model.
        omega (dict): A dictionary to store the importance weights of the parameters.
        importance (dict): A dictionary to store the accumulated importance of the parameters.
        head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.

    Methods:
        __init__(alpha: float, eps: float = 1e-6, head_opt: bool = True):
            Initializes the SI method with the given regularization strength and epsilon value.
        setup_task(task_id: int):
            Sets up the task by initializing or updating the importance weights and previous parameters.
        forward(x, y, loss, preds):
            Computes the loss with the regularization term and updates the importance weights.
        _compute_importance():
            Computes the importance of the parameters based on the changes in their values.
    """

    def __init__(self,
        alpha: float,
        eps: float = 1e-6,
        head_opt: bool = True
    ):
        """
        Initialize the instance with the given parameters.

        Args:
            alpha (float): A parameter for the method.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.
            head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.
        """
    
        super().__init__()
        self.task_id = None
        self.alpha = alpha
        self.eps = eps
        self.head_opt = head_opt
        log.info(f"Initialized SI with alpha={alpha}")

        self.prev_param = {}
        self.omega = {}
        self.importance = {}


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task_id. Initializes or updates parameters based on the task_id.

        Args:
            task_id (int): The identifier for the task. If task_id is 0, initializes parameters for a new task.
                       If task_id is greater than 0, computes the importance of the parameters for the current task.

        Attributes:
            task_id (int): Stores the current task identifier.
            prev_param (dict): Stores a copy of the parameters from the previous task.
            omega (dict): Stores the omega values for the parameters.
            importance (dict): Stores the importance values for the parameters.
        """

        self.task_id = task_id
        if task_id == 0:
            for name, p in self.module.named_parameters():
                if not self.head_opt and "head" in name:
                    continue
                if p.requires_grad:
                    self.prev_param[name] = p.data.clone().detach()
                    self.omega[name] = torch.zeros_like(p)
                    self.importance[name] = torch.zeros_like(p)
        elif task_id > 0:
            self._compute_importance()


    def forward(self, x, y, loss, preds):
        """
        Perform a forward pass and update the importance of parameters.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Computed loss.
            preds (torch.Tensor): Predictions from the model.

        Returns:
            tuple: Updated loss and predictions.
        """

        params_buffer = {}
        for name, p in self.module.named_parameters():
            if not self.head_opt and "head" in name:
                continue
            if p.requires_grad:
                params_buffer[name] = pad_zero_dim0(self.prev_param[name], p.shape)
                if p.grad is not None:
                    delta_param = p.data - params_buffer[name]
                    self.omega[name] += p.grad * (-delta_param) / (delta_param ** 2 + self.eps)
                    self.prev_param[name] = p.data.clone().detach()

        loss += self.alpha*param_change_loss(self.module, self.importance, params_buffer, self.head_opt)
        return loss, preds
    

    def _compute_importance(self):
        """
        Compute the importance of each parameter in the module.
        This method iterates over all named parameters in the module and updates their importance
        based on the difference between the current parameter values and their previous values.
        The importance is calculated using the omega values and a small epsilon to avoid division by zero.
        The importance is stored in the `self.importance` dictionary, and the omega values are reset to zero
        after the computation.
        Note:
            - `pad_zero_dim0` is a helper function that pads the tensor to match the shape of the parameter.
            - `self.omega` is a dictionary storing omega values for each parameter.
            - `self.prev_param` is a dictionary storing previous parameter values.
            - `self.importance` is a dictionary storing the computed importance for each parameter.
            - `self.eps` is a small constant to avoid division by zero.
        """

        for name, p in self.module.named_parameters():
            if not self.head_opt and "head" in name:
                continue
            if p.requires_grad:
                tmp_omega = pad_zero_dim0(self.omega[name], p.shape)
                tmp_prev = pad_zero_dim0(self.prev_param[name], p.shape)
                self.importance[name] = pad_zero_dim0(self.importance[name], p.shape)+(tmp_omega / (((p.data - tmp_prev)**2)+self.eps))
                self.omega[name] = torch.zeros_like(p)