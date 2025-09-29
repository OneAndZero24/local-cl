import logging
from copy import deepcopy
from functools import reduce
import operator

import torch

from method.regularization import param_change_loss
from src.method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class MAS(MethodPluginABC):
    """
    MAS (Memory Aware Synapses) is a method plugin for continual learning that helps mitigate catastrophic forgetting by 
    preserving important parameters of the model.

    Attributes:
        alpha (float): A hyperparameter that balances the importance of the new task loss and the parameter change loss.
        task_id (int): The identifier for the current task.
        data_buffer (set): A buffer to store data samples.
        params_buffer (dict): A buffer to store the parameters of the model.
        importance (dict): A dictionary to store the importance of each parameter.
        head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.

    Methods:
        __init__(alpha: float, head_opt: bool):
            Initializes the MAS plugin with the given alpha value.
        setup_task(task_id: int):
            Sets up the task by storing the task ID, freezing the parameters, and computing their importance.
        forward(x, y, loss, preds):
            Processes the input data, updates the data buffer, and computes the loss considering the parameter importance.
        _compute_importance():
            Computes the importance of each parameter based on the gradients of the model's outputs.
    """

    def __init__(self, 
        alpha: float,
        head_opt: bool = True
    ):
        """
        Initializes the instance of the class.

        Args:
            alpha (float): A floating-point value representing the alpha parameter.
            head_opt (bool): A flag to indicate whether EWC should be applied to the incremental head.

        Attributes:
            task_id (None): An attribute to store the task ID, initialized to None.
            alpha (float): Stores the value of the alpha parameter.
            data_buffer (set): A set to buffer data, initialized as an empty set.
            params_buffer (dict): A dictionary to buffer parameters, initialized as an empty dictionary.
            importance (dict): A dictionary to store importance values, initialized as an empty dictionary.
        """

        super().__init__()
        self.task_id = None
        self.alpha = alpha
        self.head_opt = head_opt
        log.info(f"Initialized MAS with alpha={alpha}")

        self.data_buffer = set()
        self.params_buffer = {}
        self.importance = {}


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID.

        Args:
            task_id (int): The ID of the task to set up.

        This method performs the following actions:
        - Assigns the task ID to the instance variable `self.task_id`.
        - If the task ID is greater than 0:
            - Iterates over the named parameters of the module, deep copies them, and sets `requires_grad` to False.
            - Stores these parameters in `self.params_buffer`.
            - Computes the importance of the parameters and assigns it to `self.importance`.
        - Initializes `self.data_buffer` as an empty list.
        """

        self.task_id = task_id
        if task_id > 0:
            for name, p in deepcopy(list(self.module.named_parameters())):
                if not self.head_opt and "head" in name:
                    continue
                if p.requires_grad:
                    p.requires_grad = False
                    self.params_buffer[name] = p    
            self.importance = self._compute_importance()
        self.data_buffer = set()


    def forward(self, x, y, loss, preds):
        """
        Perform a forward pass and compute the loss.

        Args:
            x (Tensor): Input data.
            y (Tensor): Target labels.
            loss (Tensor): Initial loss value.
            preds (Tensor): Predictions from the model.

        Returns:
            Tuple[Tensor, Tensor]: Updated loss and predictions.
        """

        self.data_buffer.add((x, y))

        if self.task_id > 0:
            loss += self.alpha*param_change_loss(self.module, self.importance, self.params_buffer, self.head_opt)
        return loss, preds
    

    def _compute_importance(self):
        """
        Compute the importance of each parameter in the module based on the gradients.
        This method evaluates the module, iterates over the data buffer, and computes the 
        importance of each parameter by accumulating the absolute value of the gradients 
        multiplied by the number of inputs. The importance is then averaged over the 
        number of batches in the data buffer.
        
        Returns:
            dict: A dictionary where keys are parameter names and values are tensors 
                  representing the importance of each parameter.
        """

        self.module.eval()
        importance = {name: torch.zeros_like(param) for name, param in self.module.named_parameters() if param.requires_grad}
        
        for inputs, _ in self.data_buffer:
            self.module.zero_grad()
            outputs = self.module(inputs)
            loss = (outputs.norm(2) ** 2)/reduce(operator.mul, outputs.shape[1:])
            loss.backward()
            
            for name, param in self.module.named_parameters():
                if not self.head_opt and "head" in name:
                    continue
                if param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.abs() * len(inputs)

        for name in importance:
            importance[name] /= len(self.data_buffer)
        
        return importance