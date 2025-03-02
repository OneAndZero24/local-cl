import logging

import torch

from src.method.method_plugin_abc import MethodPluginABC
from method.regularization import sharpen_loss
from util.tensor import pad_zero_dim0


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Sharpening(MethodPluginABC):
    """
    Sharpening is a method plugin that adjusts activations by additional backward increasing most active nodes.
        https://cdn.aaai.org/Symposia/Spring/1993/SS-93-06/SS93-06-007.pdf

    Attributes:
        alpha (float): The scaling factor for the original loss.
        gamma (float): The scaling factor for the sharpening loss.
        K (int): The number of top activations to consider for sharpening.

    Methods:
        setup_task(task_id: int):
            Placeholder method for setting up a task. Currently does nothing.

        forward(x, y, loss, preds):
            Adjusts the loss by combining the original loss with a sharpening loss.
    """

    def __init__(self, 
        alpha: float,
        gamma: float,
        K : int
    ):
        """
        Initializes the sharpening method with the given parameters.

        Args:
            alpha (float): The alpha parameter for the sharpening method.
            gamma (float): The gamma parameter for the sharpening method.
            K (int): The K parameter for the sharpening method.
        """
                
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.K = K
        log.info(f"Initialized Sharpening with alpha={alpha}, gamma={gamma}, K={K}")

        self.activation_buffer = None


    def setup_task(self, task_id: int):
        """
        Sets up a task with the given task ID.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """

        self.activation_buffer = None


    def forward(self, x, y, loss, preds):
        """
        Perform the forward pass of the sharpening method.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Initial loss value.
            preds (torch.Tensor): Predictions tensor.
            
        Returns:
            tuple: Updated loss and predictions tensors.
        """

        activations = torch.cat(self.module.activations, dim=1).sum(dim=0)/x.shape[0]
        activations_det = activations.clone().detach_()
        activations_det.requires_grad = False
        if self.activation_buffer is None:
            self.activation_buffer = torch.zeros_like(activations_det)
        tmp = pad_zero_dim0(activations_det, self.activation_buffer.shape)
        self.activation_buffer += tmp

        _, indices = torch.topk(self.activation_buffer.view(-1), self.K)

        loss *= self.alpha
        loss += (1-self.alpha)*sharpen_loss(indices, tmp, self.gamma)
        return loss, preds