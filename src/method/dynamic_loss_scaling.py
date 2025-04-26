import logging
import torch
import math

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DynamicScaling:
    """
    Dynamic Scaling for Continual Learning.

    This class dynamically scales the cross-entropy (CE) loss using a computed 
    lambda factor based on the L2-norm difference between CE and regularization 
    gradients, with Huber-weighted gradients, adaptive EMA smoothing using TMAD, 
    variance-weighted normalization, polynomial scaling to ensure dynamic_lambda = 1 
    at angle 0° and 0 at angle 180°, and running variance for lambda smoothing, 
    reducing seed variability and handling noisy gradients. Entropy scaling is disabled.

    Attributes:
        module (torch.nn.Module): The neural network model.
        min_lambda (float): Minimum value of lambda.
        max_lambda (float): Maximum value of lambda.
        beta (float): Exponent for polynomial scaling.
        ema_scale_base (float): Base EMA factor for smoothing lambda updates.
        grad_ema_scale_ce_base (float): Base EMA factor for smoothing CE gradients.
        grad_ema_scale_reg_base (float): Base EMA factor for smoothing regularization gradients.
        huber_delta_scale (float): Scale factor for Huber weighting (TMAD-based).
        prev_dynamic_lambda (float or None): Previous lambda value for EMA.
        prev_grads_ce (list or None): Smoothed CE gradients.
        prev_grads_reg (list or None): Smoothed regularization gradients.
        alignment_var (float or None): Running variance of alignment.
    """

    def __init__(self, module, min_lambda: float, max_lambda: float, beta: float, 
                 ema_scale_base: float, grad_ema_scale_ce_base: float, grad_ema_scale_reg_base: float,
                 huber_delta_scale: float = 1.5):
        """
        Initializes the DynamicScaling module.

        Args:
            module (torch.nn.Module): The neural network model.
            min_lambda (float): Minimum lambda value.
            max_lambda (float): Maximum lambda value.
            beta (float): Exponent for polynomial scaling.
            ema_scale_base (float): Base EMA factor for smoothing lambda updates.
            grad_ema_scale_ce_base (float): Base EMA factor for smoothing CE gradients.
            grad_ema_scale_reg_base (float): Base EMA factor for smoothing regularization gradients.
            huber_delta_scale (float): Scale factor for Huber weighting (TMAD-based).
        """
        super().__init__()
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.beta = beta
        self.ema_scale = ema_scale_base
        self.grad_ema_scale_ce_base = grad_ema_scale_ce_base
        self.grad_ema_scale_reg_base = grad_ema_scale_reg_base
        self.huber_delta_scale = huber_delta_scale
        self.module = module

        self.prev_dynamic_lambda = None
        self.prev_grads_ce = None
        self.prev_grads_reg = None
        self.alignment_var = None

    def forward(self, task_id: int, loss_ce: torch.Tensor, loss_reg: torch.Tensor, 
                preds: torch.Tensor) -> torch.Tensor:
        """
        Computes the dynamically scaled loss based on gradient alignment.

        Args:
            task_id (int): Current task index.
            loss_ce (torch.Tensor): Cross-entropy loss.
            loss_reg (torch.Tensor): Regularization loss.
            preds (torch.Tensor): Model predictions (unused, kept for compatibility).

        Returns:
            torch.Tensor: Weighted sum of loss_ce and loss_reg.
        """
        if task_id > 0 and self.module.training:
            grads_reg = torch.autograd.grad(loss_reg, self.module.parameters(), retain_graph=True)            
            grads_ce = torch.autograd.grad(loss_ce, self.module.parameters(), retain_graph=True)
            dynamic_lambda = self.compute_dynamic_lambda(grads_ce, grads_reg)
        else:
            dynamic_lambda = 1.0
        return dynamic_lambda * loss_ce + loss_reg
    
    def _huber_weights(self, grad: torch.Tensor, delta_scale: float, eps: float=1e-6) -> torch.Tensor:
        """
        Computes Huber weights for gradient components to downweight outliers.

        Args:
            grad (torch.Tensor): Gradient tensor.
            delta_scale (float): Scale factor for Huber delta.
            eps (float): Small value for numerical stability.

        Returns:
            torch.Tensor: Weights for gradient components.
        """
        abs_grad = torch.abs(grad)
        mad = torch.median(abs_grad) + eps
        delta = delta_scale * mad
        weights = torch.where(abs_grad <= delta, torch.ones_like(grad), delta / (abs_grad + eps))
        return weights

    def _alignment_score(self, x1: torch.Tensor, x2: torch.Tensor, weights: torch.Tensor,
                        eps: float=1e-6) -> torch.Tensor:
        """
        Computes the alignment score using weighted L2-norm difference between normalized gradients.

        Args:
            x1 (torch.Tensor): First gradient tensor.
            x2 (torch.Tensor): Second gradient tensor.
            weights (torch.Tensor): Weights for gradient components (inverse variance).
            eps (float): Small value for numerical stability.

        Returns:
            torch.Tensor: Alignment score in [0, 1].
        """
        x1_norm = x1 / (torch.norm(x1, dim=-1, keepdim=True) + eps)
        x2_norm = x2 / (torch.norm(x2, dim=-1, keepdim=True) + eps)
        diff = x1_norm - x2_norm
        l2_diff = torch.sqrt(torch.sum(weights * diff**2, dim=-1) / (torch.sum(weights) + eps))
        alignment = 1.0 - l2_diff / 2.0
        return alignment.clamp(0.0, 1.0)

    def compute_dynamic_lambda(self, grads_ce: list, grads_reg: list) -> float:
        """
        Computes dynamic_lambda using L2-norm difference, polynomial scaling, Huber-weighted 
        gradients, adaptive EMA with TMAD, and variance-weighted normalization.

        Ensures dynamic_lambda = 1 at alignment = 1 (angle 0°) and 0 at alignment = 0 (angle 180°).

        Args:
            grads_ce (list of torch.Tensor): CE loss gradients.
            grads_reg (list of torch.Tensor): Regularization loss gradients.

        Returns:
            float: Updated lambda_t value in [min_lambda, max_lambda].
        """
        # Apply Huber weights
        grads_ce = [g * self._huber_weights(g, self.huber_delta_scale) for g in grads_ce if g is not None]
        grads_reg = [g * self._huber_weights(g, self.huber_delta_scale) for g in grads_reg if g is not None]

        # Smooth gradients using adaptive and iterative EMA
        if self.prev_grads_ce is None or self.prev_grads_reg is None:
            smoothed_grads_ce = [g.clone() for g in grads_ce if g is not None]
            smoothed_grads_reg = [g.clone() for g in grads_reg if g is not None]
            self.prev_grads_ce = smoothed_grads_ce
            self.prev_grads_reg = smoothed_grads_reg
        else:
            smoothed_grads_ce = [
                self.grad_ema_scale_ce_base * prev_g + (1 - self.grad_ema_scale_ce_base) * g
                for prev_g, g in zip(self.prev_grads_ce, grads_ce) if g is not None
            ]
            smoothed_grads_reg = [
                self.grad_ema_scale_reg_base * prev_g + (1 - self.grad_ema_scale_reg_base) * g
                for prev_g, g in zip(self.prev_grads_reg, grads_reg) if g is not None
            ]
            self.prev_grads_ce = smoothed_grads_ce
            self.prev_grads_reg = smoothed_grads_reg

        # Flatten gradients
        grads_ce_flat = torch.cat([g.flatten() for g in smoothed_grads_ce if g is not None], dim=0)
        grads_reg_flat = torch.cat([g.flatten() for g in smoothed_grads_reg if g is not None], dim=0)

        if grads_ce_flat.numel() == 0 or grads_reg_flat.numel() == 0:
            log.warning("Skipping lambda_t update due to missing gradients.")
            return self.prev_dynamic_lambda if self.prev_dynamic_lambda is not None else self.min_lambda

        if not (torch.isfinite(grads_ce_flat).all() and torch.isfinite(grads_reg_flat).all()):
            log.warning("Skipping lambda_t update due to invalid gradients.")
            return self.prev_dynamic_lambda if self.prev_dynamic_lambda is not None else self.min_lambda

        # Compute variance weights
        var_ce = torch.var(grads_ce_flat, unbiased=False) + 1e-6
        var_reg = torch.var(grads_reg_flat, unbiased=False) + 1e-6
        weights = 1.0 / (var_ce + var_reg)

        # Compute alignment score
        alignment = self._alignment_score(
            grads_ce_flat.unsqueeze(0), 
            grads_reg_flat.unsqueeze(0),
            weights).item()

        # Polynomial scaling
        dynamic_lambda = alignment ** self.beta

        # Apply EMA to dynamic_lambda
        if self.prev_dynamic_lambda is None:
            self.prev_dynamic_lambda = self.min_lambda
        else:
            dynamic_lambda = self.ema_scale * dynamic_lambda + (1 - self.ema_scale) * self.prev_dynamic_lambda
            self.prev_dynamic_lambda = dynamic_lambda

        # Clamp to [min_lambda, max_lambda]
        return max(self.min_lambda, min(dynamic_lambda, self.max_lambda))