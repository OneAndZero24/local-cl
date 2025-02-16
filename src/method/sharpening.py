from copy import deepcopy

import torch

from method.regularization import distillation_loss
from src.method.method_plugin_abc import MethodPluginABC


class Sharpening(MethodPluginABC):
    """
    Sharpening is a method plugin that adjusts activations by additional backward increasing most active nodes.
    https://cdn.aaai.org/Symposia/Spring/1993/SS-93-06/SS93-06-007.pdf

    Attributes:
        alpha (float): The sharpening factor.
        K (int): The number of top activations to sharpen.
        task_id (int): The ID of the current task.
        N (int): The number of forward passes.
        activations_buffer (torch.Tensor): Buffer to store activations.

    Methods:
        __init__(alpha: float, K: int):
            Initializes the Sharpening plugin with the given alpha and K values.
        setup_task(task_id: int):
            Sets up the task with the given task ID. Adjusts activations if the task ID is greater than 0.
        forward(x, y, loss, preds):
            Processes the input data, updates the activations buffer, and returns the loss and predictions.
    """
    def __init__(self, 
        alpha: float,
        K : int,
        M : int = 1
    ):
        """
        Initialize the sharpening method.

        Args:
            alpha (float): The alpha parameter for the sharpening method.
            K (int): The K parameter for the sharpening method.
            M (int): Log activations every M steps. Default is 1.
        """

        super().__init__()
        self.task_id = None
        self.N = 0
        self.activations_buffer = None

        self.alpha = alpha
        self.K = K
        self.M = M


    def setup_task(self, task_id: int):
        """
        Sets up the task with the given task ID. If the task ID is greater than 0, it performs
        a sharpening operation on the activations buffer by adjusting the top K activations 
        and applying a backward pass with the computed difference.

        Args:
            task_id (int): The ID of the task to set up.

        Attributes:
            task_id (int): The ID of the task.
            activations_buffer (torch.Tensor): The buffer containing activations to be processed.
            N (int): A normalization factor.
            K (int): The number of top activations to adjust.
            alpha (float): The sharpening factor.
            composer (object): An object with a backward method to apply the computed difference.
        """

        self.task_id = task_id

        if task_id > 0:         
            self.activations_buffer /= self.N
            flattened_activations = self.activations_buffer.view(-1)
            _, indices = torch.topk(flattened_activations, self.K)
            
            new_activations = self.activations_buffer.clone()
            mask = torch.zeros_like(flattened_activations, dtype=torch.bool)
            mask[indices] = True
            
            self.activations_buffer.view(-1)[mask] += self.alpha * (1 - self.activations_buffer.view(-1)[mask])
            self.activations_buffer.view(-1)[~mask] -= self.alpha * self.activations_buffer.view(-1)[~mask]

            diff = torch.sum(torch.square(new_activations - self.activations_buffer))
            self.composer.backward(diff)

        self.N = 0
        self.activations_buffer = None


    def forward(self, x, y, loss, preds):
        """
        Forward pass for the sharpening method.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Loss value.
            preds (torch.Tensor): Predictions tensor.
            
        Returns:
            tuple: A tuple containing the loss and predictions tensors.
        """

        self.N += 1
        if (self.N % self.M) == 0:
            activations_t = torch.stack(self.module.activations)
            if self.activations_buffer is None:
                self.activations_buffer = activations_t
            else:
                self.activations_buffer[tuple(slice(0, s) for s in activations_t.shape)] += activations_t
        return loss, preds