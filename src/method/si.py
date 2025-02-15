from copy import deepcopy
import torch

from method.regularization import param_change_loss
from src.method.method_plugin_abc import MethodPluginABC


class SI(MethodPluginABC): # TODO Fix & Cleanup
    """
    SI (Synaptic Intelligence) is a method plugin for continual learning that helps mitigate catastrophic forgetting by 
    regularizing the changes in the network's parameters. It does so by maintaining an importance measure for each parameter 
    and penalizing significant changes to important parameters.

    Attributes:
        alpha (float): Regularization strength parameter.
        eps (float): Small value to avoid division by zero.
        task_id (int): Identifier for the current task.
        data_buffer (list): Buffer to store data samples.
        params_buffer (dict): Buffer to store parameters of the model.
        importance (dict): Importance measure for each parameter.
        omega (dict): Accumulated importance measure for each parameter.

    Methods:
        __init__(alpha: float, eps: float = 1e-6):
            Initializes the SI plugin with the given regularization strength and epsilon value.
        setup_task(task_id: int):
            Sets up the task by updating the task identifier, computing the importance of parameters, and resetting the data buffer.
        forward(x, y, loss, preds):
            Processes the input data, updates the data buffer, and computes the loss with regularization if necessary.
        _compute_importance():
            Computes the importance of each parameter based on the accumulated omega values and the parameter changes.
    """

    def __init__(self,
        alpha: float,
        eps: float = 1e-6
    ):
        """
        Initialize the instance with given parameters.

        Args:
            alpha (float): A coefficient parameter used in the method.
            eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.

        Attributes:
            task_id (None): Identifier for the current task, initialized to None.
            alpha (float): A coefficient parameter used in the method.
            eps (float): A small value to prevent division by zero.
            data_buffer (list): A buffer to store data.
            params_buffer (dict): A buffer to store parameters.
            importance (dict): A dictionary to store importance values.
            omega (dict): A dictionary to store omega values for each parameter.
        """
    
        super().__init__()
        self.task_id = None
        self.alpha = alpha
        self.eps = eps

        self.data_buffer = []
        self.params_buffer = {}
        self.importance = {}
        self.omega = {}

    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID. This method updates the model's parameters
        and computes their importance based on the gradients.

        Args:
            task_id (int): The ID of the task to set up.

        Side Effects:
            - Updates the `task_id` attribute.
            - Updates the `params_buffer` with the current parameters.
            - Computes and updates the `omega` attribute with the importance of each parameter.
            - Sets `requires_grad` to False for all parameters.
            - Computes and updates the `importance` attribute.
            - Clears the `data_buffer` list.

        Notes:
            - This method assumes that `self.module.named_parameters()` returns the model's parameters.
            - The `self.params_buffer` should be a dictionary containing the previous parameters.
            - The `self.omega` should be a dictionary to store the importance of each parameter.
            - The `self.eps` is a small value to prevent division by zero.
        """

        self.task_id = task_id
        if task_id == 0:
            for name, p in self.module.named_parameters():
                if p.requires_grad:
                    self.omega[name] = torch.zeros_like(p)
        elif task_id > 0:
            for name, p in deepcopy(list(self.module.named_parameters())):
                if p.requires_grad and p.grad is not None:
                    delta_param = p.data - self.params_buffer[name].data
                    self.omega[name] += p.grad * delta_param / (delta_param ** 2 + self.eps)  
                p.requires_grad = False
                self.params_buffer[name] = p
            self.importance = self._compute_importance()
        self.data_buffer = []


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

        self.data_buffer.append((x, y))

        if self.task_id > 0:
            loss *= self.alpha
            loss += (1-self.alpha)*param_change_loss(self.module, self.importance, self.params_buffer)
        return loss, preds
    

    def _compute_importance(self):
        """
        Compute the importance of each parameter in the module.
        This method calculates the importance of each parameter by iterating over 
        the named parameters of the module. For each parameter that requires a gradient, 
        it updates the importance dictionary with the computed importance value, which 
        is based on the omega value and the difference between the current parameter 
        value and its buffered value. The omega values are then reset to zero.

        Returns:
            dict: A dictionary where keys are parameter names and values are tensors 
                  representing the importance of each parameter.
        """

        importance = {name: torch.zeros_like(param) for name, param in self.module.named_parameters()}
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                importance[name] += self.omega[name] / ((p.data - self.params_buffer[name])**2) + self.eps
                self.omega[name].zero_()
        return importance