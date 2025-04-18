import logging
import torch
from torch.nn import functional as F


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DynamicScaling():
    """
    Dynamic Scaling for Continual Learning.

    This class dynamically scales the cross-entropy (CE) loss using a computed 
    lambda factor based on gradient alignment between CE loss and regularization loss.

    The scaling factor is determined by an exponential annealing approach 
    that prevents conflicting gradient directions and smooths updates using 
    an Exponential Moving Average (EMA).

    Attributes:
        module (torch.nn.Module): The neural network model.
        min_lambda (float): Minimum value of lambda (lower bound).
        max_lambda (float): Maximum value of lambda (upper bound).
        beta (float): Scaling factor that controls the rate of exponential annealing.
        ema_scale (float): Exponential Moving Average (EMA) factor for smoothing updates.
        prev_dynamic_lambda (float or None): Stores the previous lambda value for EMA updates.
    """

    def __init__(self, module, min_lambda: float, max_lambda: float, beta: float, ema_scale: float):
        """
        Initializes the DynamicScaling module.

        Args:
            module (torch.nn.Module): The neural network model.
            min_lambda (float): Minimum lambda value for scaling CE loss.
            max_lambda (float): Maximum lambda value for scaling CE loss.
            beta (float): Exponential decay factor controlling the lambda update.
            ema_scale (float): EMA factor for smoothing lambda updates.

        Methods:
            __init__(module, min_lambda: float, max_lambda: float, beta: float, ema_scale: float):
                Initializes the DynamicScaling method with the given hyperparameters.
            forward(task_id, loss_ce, loss_reg):
                Computes the dynamically scaled loss by adjusting the cross-entropy loss weight.
            compute_dynamic_lambda(grads_ce, grads_reg):
                Computes the lambda scaling factor based on gradient alignment.
        """
        super().__init__()
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.beta = beta
        self.ema_scale = ema_scale
        self.module = module

        self.prev_dynamic_lambda = None

    def forward(self, task_id: int, loss_ce: torch.Tensor, loss_reg: torch.Tensor, 
                preds: torch.Tensor) -> torch.Tensor:
        """
        Computes the dynamically scaled loss based on gradient alignment.

        The method dynamically adjusts the weighting of the cross-entropy (CE) 
        loss based on its alignment with the regularization loss.

        Args:
            task_id (int): The current task index in a continual learning setup.
            loss_ce (torch.Tensor): Cross-entropy loss for classification.
            loss_reg (torch.Tensor): Regularization loss (e.g., EWC, L2 penalty).
            preds (torch.Tensor): Model predictions.

        Returns:
            torch.Tensor: The weighted sum of `loss_ce` and `loss_reg`, 
                          where `loss_ce` is scaled dynamically.
        """
        if task_id > 0 and self.module.training:
            grads_reg = torch.autograd.grad(loss_reg, self.module.parameters(), retain_graph=True)            
            grads_ce = torch.autograd.grad(loss_ce, self.module.parameters(), retain_graph=True)
            dynamic_lambda = self.compute_dynamic_lambda(grads_ce, grads_reg, preds)
        else:
            dynamic_lambda = 1.0
        return dynamic_lambda * loss_ce + loss_reg

    def compute_dynamic_lambda(self, grads_ce, grads_reg, preds):
        """
        Computes the dynamic lambda_t using exponential annealing of misaligned gradients.

        This method calculates the alignment between gradients of the CE loss and 
        the regularization loss using cosine similarity. The lambda scaling factor is 
        updated accordingly to ensure that the optimization does not move in conflicting 
        gradient directions.

        The update process is stabilized using an Exponential Moving Average (EMA).

        Args:
            grads_ce (list of torch.Tensor): Gradients of cross-entropy loss.
            grads_reg (list of torch.Tensor): Gradients of regularization loss.
            preds (torch.Tensor): Model predictions.

        Returns:
            float: The updated lambda_t value, ensuring it remains within 
                   `[min_lambda, max_lambda]`.
        """

        grads_ce_flat = torch.cat([g.flatten() for g in grads_ce if g is not None], dim=0)
        grads_reg_flat = torch.cat([g.flatten() for g in grads_reg if g is not None], dim=0)

        if grads_ce_flat.numel() == 0 or grads_reg_flat.numel() == 0:
            log.warning("Skipping lambda_t update due to missing gradients.")
            return self.prev_dynamic_lambda 

        cos_theta = F.cosine_similarity(
            grads_ce_flat.unsqueeze(0), 
            grads_reg_flat.unsqueeze(0), dim=1, eps=1e-8
        ).item()

        preds = F.softmax(preds, dim=-1)
        entropy = -torch.sum(preds * preds.log(), dim=1).mean().item()
        entropy /= torch.log(torch.tensor(preds.size(1))).item()

        dynamic_lambda = torch.exp(torch.tensor(-self.beta * (1 - cos_theta))).item()
        dynamic_lambda *= entropy

        if self.prev_dynamic_lambda is None:
            self.prev_dynamic_lambda = self.min_lambda
        else:
            dynamic_lambda = self.ema_scale * dynamic_lambda + (1 - self.ema_scale) * self.prev_dynamic_lambda
            self.prev_dynamic_lambda = dynamic_lambda

        return max(self.min_lambda, min(dynamic_lambda, self.max_lambda))