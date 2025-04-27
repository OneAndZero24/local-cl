import logging
import torch
import math

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DynamicScaling:
    """
    Dynamic Scaling for Continual Learning, with Stable-Reference Alignment.

    This class dynamically scales the cross-entropy (CE) loss relative to a 
    stable regularization gradient (grad_reg). It applies Huber weighting to noisy CE gradients, 
    exponential moving average (EMA) smoothing for CE gradients, and computes 
    alignment via a projection-based stable reference method.

    Dynamic lambda is discretized into bins to improve robustness against noisy estimates.
    """

    def __init__(self, module, min_lambda: float,
                 ema_scale_base: float, grad_ema_scale_ce_base: float,
                 huber_delta_scale: float = 1.5):
        """
        Initializes the DynamicScaling module.

        Args:
            module (torch.nn.Module): The neural network model.
            min_lambda (float): Minimum value of dynamic lambda.
            ema_scale_base (float): EMA smoothing factor for dynamic lambda updates.
            grad_ema_scale_ce_base (float): EMA smoothing factor for CE gradients.
            huber_delta_scale (float, optional): Scale factor for Huber weighting (default=1.5).
        """
        super().__init__()
        self.min_lambda = min_lambda
        self.ema_scale = ema_scale_base
        self.grad_ema_scale_ce_base = grad_ema_scale_ce_base
        self.huber_delta_scale = huber_delta_scale
        self.module = module

        self.prev_dynamic_lambda = None
        self.prev_grads_ce = None

    def forward(self, task_id: int, loss_ce: torch.Tensor, loss_reg: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """
        Computes the dynamically scaled loss based on current task and training mode.

        Args:
            task_id (int): Current task ID.
            loss_ce (torch.Tensor): Cross-entropy loss.
            loss_reg (torch.Tensor): Regularization loss.
            preds (torch.Tensor): Model predictions (unused, for interface compatibility).

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
        Computes Huber weights to downweight outlier components of the gradient.

        Args:
            grad (torch.Tensor): Gradient tensor.
            delta_scale (float): Scaling factor for Huber threshold.
            eps (float, optional): Small epsilon for numerical stability.

        Returns:
            torch.Tensor: Weights tensor.
        """
        abs_grad = torch.abs(grad)
        mad = torch.median(abs_grad) + eps
        delta = delta_scale * mad
        weights = torch.where(abs_grad <= delta, torch.ones_like(grad), delta / (abs_grad + eps))
        return weights

    def _alignment_score_stable_ref(self, grads_ce_flat: torch.Tensor, grads_reg_flat: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
        """
        Computes alignment score by projecting CE gradients onto stable regularization gradients.

        Args:
            grads_ce_flat (torch.Tensor): Flattened CE gradients.
            grads_reg_flat (torch.Tensor): Flattened regularization gradients.
            eps (float, optional): Small epsilon for numerical stability.

        Returns:
            torch.Tensor: Alignment score in [0, 1].
        """
        proj = (grads_ce_flat * grads_reg_flat).sum() / (grads_reg_flat.norm()**2 + eps)
        residual = grads_ce_flat - proj * grads_reg_flat
        residual_norm = residual.norm()
        ce_norm = grads_ce_flat.norm()
        alignment = 1.0 - (residual_norm / (ce_norm + eps))
        return alignment.clamp(0.0, 1.0)

    def compute_dynamic_lambda(self, grads_ce: list, grads_reg: list) -> float:
        """
        Computes the dynamic lambda based on alignment between smoothed CE gradients and true REG gradients.
        
        Applies Huber weighting, EMA smoothing, stable-reference alignment, polynomial scaling, 
        and binning into discrete steps to stabilize dynamic lambda.

        Args:
            grads_ce (list of torch.Tensor): CE loss gradients.
            grads_reg (list of torch.Tensor): Regularization loss gradients.

        Returns:
            float: Updated dynamic lambda value.
        """
        grads_ce = [g * self._huber_weights(g, self.huber_delta_scale) for g in grads_ce if g is not None]
        grads_reg = [g for g in grads_reg if g is not None]

        # EMA smoothing only for CE gradients
        if self.prev_grads_ce is None:
            smoothed_grads_ce = [g.clone() for g in grads_ce]
            self.prev_grads_ce = smoothed_grads_ce
        else:
            smoothed_grads_ce = [
                self.grad_ema_scale_ce_base * prev_g + (1 - self.grad_ema_scale_ce_base) * g
                for prev_g, g in zip(self.prev_grads_ce, grads_ce)
            ]
            self.prev_grads_ce = smoothed_grads_ce

        grads_ce_flat = torch.cat([g.flatten() for g in smoothed_grads_ce], dim=0)
        grads_reg_flat = torch.cat([g.flatten() for g in grads_reg], dim=0)

        if grads_ce_flat.numel() == 0 or grads_reg_flat.numel() == 0:
            log.warning("Skipping dynamic_lambda update due to missing gradients.")
            return self.prev_dynamic_lambda if self.prev_dynamic_lambda is not None else self.min_lambda

        if not (torch.isfinite(grads_ce_flat).all() and torch.isfinite(grads_reg_flat).all()):
            log.warning("Skipping dynamic_lambda update due to invalid gradients.")
            return self.prev_dynamic_lambda if self.prev_dynamic_lambda is not None else self.min_lambda

        dynamic_lambda = self._alignment_score_stable_ref(grads_ce_flat, grads_reg_flat).item()

        # EMA smoothing on lambda itself
        if self.prev_dynamic_lambda is None:
            self.prev_dynamic_lambda = dynamic_lambda
        else:
            dynamic_lambda = self.ema_scale * dynamic_lambda + (1 - self.ema_scale) * self.prev_dynamic_lambda
            self.prev_dynamic_lambda = dynamic_lambda

        return max(self.min_lambda, dynamic_lambda)
