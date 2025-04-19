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
        angle_constraint_scale (float): A factor that adjusts the influence of angular constraints 
                                        when computing the dynamic scaling factor.
    """

    def __init__(self, module, min_lambda: float, max_lambda: float, beta: float, 
                 ema_scale: float, use_entropy_scale: bool, angle_constraint_scale: float):
        """
        Initializes the DynamicScaling module.

        Args:
            module (torch.nn.Module): The neural network model.
            min_lambda (float): Minimum lambda value for scaling CE loss.
            max_lambda (float): Maximum lambda value for scaling CE loss.
            beta (float): Exponential decay factor controlling the lambda update.
            ema_scale (float): EMA factor for smoothing lambda updates.
            use_entropy_scale (bool): Flag to indicate if entropy scaling should be used.
            angle_constraint_scale (float): A factor that adjusts the influence of angular constraints 
                                        when computing the dynamic scaling factor.

        Methods:
            __init__(module, min_lambda: float, max_lambda: float, beta: float, ema_scale: float, 
                use_entropy_scale: bool, angle_constraint_scale: float):
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
        self.use_entropy_scale = use_entropy_scale
        self.angle_constraint_scale = angle_constraint_scale

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
    
    def _cosine_similarity(self, x1: torch.Tensor, x2: torch.Tensor, dim: int=1, 
                           eps: float=1e-8) -> torch.Tensor:
        """
        Computes a scaled cosine similarity between two input tensors.
        This function calculates the cosine similarity between two tensors, 
        scales the angle (in radians) by a given factor `alpha`, and then 
        computes the cosine of the scaled angle.
        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.
            dim (int, optional): The dimension along which the cosine similarity 
                is computed. Default is 1.
            eps (float, optional): A small value to avoid division by zero 
                during normalization. Default is 1e-8.
        Returns:
            torch.Tensor: A tensor containing the scaled cosine similarity 
            values, with the same shape as the input tensors along the 
            specified dimension.
        """

        cos_theta = F.cosine_similarity(x1, x2, dim=dim, eps=eps)        
        theta = torch.acos(cos_theta)  # theta in [0, pi]
        alpha_theta = self.angle_constraint_scale * theta        
        cos_alpha_theta = torch.cos(alpha_theta)
        
        return cos_alpha_theta

    def compute_dynamic_lambda(self, grads_ce: torch.Tensor, grads_reg: torch.Tensor, 
                               preds: torch.Tensor) -> float:
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

        cos_theta = self._cosine_similarity(
            grads_ce_flat.unsqueeze(0), 
            grads_reg_flat.unsqueeze(0)).item()

        dynamic_lambda = torch.exp(torch.tensor(-self.beta * (1 - cos_theta))).item()
        dynamic_lambda -= torch.exp(torch.tensor(-2.0 * self.beta)).item()

        if self.use_entropy_scale:
            preds = F.softmax(preds, dim=-1)
            preds = torch.clamp(preds, min=1e-8, max=1.0)
            entropy = -torch.sum(preds * preds.log(), dim=1).mean().item()
            entropy /= torch.log(torch.tensor(preds.size(1))).item()
            dynamic_lambda *= entropy
        if self.prev_dynamic_lambda is None:
            self.prev_dynamic_lambda = self.min_lambda
        else:
            dynamic_lambda = self.ema_scale * dynamic_lambda + (1 - self.ema_scale) * self.prev_dynamic_lambda
            self.prev_dynamic_lambda = dynamic_lambda

        return max(self.min_lambda, min(dynamic_lambda, self.max_lambda))