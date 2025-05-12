import logging
import torch
import wandb

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DynamicScaling:
    """
    Dynamic Scaling for Continual Learning, with Stable-Reference Alignment.

    This class dynamically scales the current task loss relative to a 
    stable regularization gradient (grad_reg). It computes alignment via a
    projection-based stable reference method.
    """

    def __init__(self, module, ema_scale_base: float, beta: float, clamp: bool = False):
        """
        Initializes the DynamicScaling module.

        Args:
            module (torch.nn.Module): The neural network model.
            ema_scale_base (float): The EMA smoothing factor for dynamic lambda updates.
            beta (float): A hyperparameter controlling vanishing of a tanh argument.
        """
        super().__init__()
        self.ema_scale = ema_scale_base
        self.beta = beta
        self.module = module
        self.clamp = clamp

        self.prev_dynamic_lambda = 0.0

    def forward(self, task_id: int, loss_ct: torch.Tensor, loss_reg: torch.Tensor,
                 preds: torch.Tensor) -> torch.Tensor:
        """
        Computes the dynamically scaled loss based on current task and training mode.

        Args:
            task_id (int): Current task ID.
            loss_ct (torch.Tensor): Cross-entropy loss.
            loss_reg (torch.Tensor): Regularization loss.
            preds (torch.Tensor): Model predictions (unused, for interface compatibility).

        Returns:
            torch.Tensor: Weighted sum of loss_ct and loss_reg.
        """
        if task_id > 0 and self.module.training:
            params = [p for p in self.module.parameters() if p.requires_grad]
            grads_reg = torch.autograd.grad(loss_reg, params, retain_graph=True)
            grads_ct = torch.autograd.grad(loss_ct, params, retain_graph=True)
            dynamic_lambda = self.compute_dynamic_lambda(grads_ct, grads_reg)
        else:
            dynamic_lambda = 1.0
        wandb.log({"dynamic_lambda": dynamic_lambda})
        return dynamic_lambda * loss_ct + loss_reg

    def _alignment_score_stable_ref(self, grads_ct_flat: torch.Tensor, grads_reg_flat: torch.Tensor, 
                                    eps: float=1e-8) -> torch.Tensor:
        """
        Computes alignment score by projecting current task loss gradients onto stable regularization gradients.

        Args:
            grads_ct_flat (torch.Tensor): Flattened, non-reduced, current task loss gradients.
            grads_reg_flat (torch.Tensor): Flattened regularization gradients.
            eps (float, optional): Small epsilon for numerical stability.

        Returns:
            torch.Tensor: Alignment score in [0, 1].
        """
        eps = torch.tensor(eps)
        proj = (grads_ct_flat * grads_reg_flat).sum() / (torch.max(grads_reg_flat.norm()**2, eps))
        alignment = self.beta * torch.sigmoid(proj / self.beta)
        wandb.log({"alignment": alignment.item()})
        wandb.log({"grads_reg_flat_norm": grads_reg_flat.norm().item()})
        wandb.log({"grads_ct_flat_norm": grads_ct_flat.norm().item()})
        if self.clamp:
            alignment = torch.clamp(alignment, min=0.0, max=1.0)
        return alignment

    def compute_dynamic_lambda(self, grads_ct: list, grads_reg: list) -> float:
        """
        Computes the dynamic lambda based on alignment between smoothed current task loss
        gradients and true REG gradients.
        
        Args:
            grads_ct (list of torch.Tensor): current task loss gradients.
            grads_reg (list of torch.Tensor): Regularization loss gradients.

        Returns:
            float: Updated dynamic lambda value.
        """
        grads_ct = [g for g in grads_ct if g is not None]
        grads_reg = [g for g in grads_reg if g is not None]

        grads_ct_flat = torch.cat([g.flatten() for g in grads_ct], dim=0)
        grads_reg_flat = torch.cat([g.flatten() for g in grads_reg], dim=0)

        if grads_ct_flat.numel() == 0 or grads_reg_flat.numel() == 0:
            log.warning("Skipping dynamic_lambda update due to missing gradients.")
            return self.prev_dynamic_lambda

        if not (torch.isfinite(grads_ct_flat).all() and torch.isfinite(grads_reg_flat).all()):
            log.warning("Skipping dynamic_lambda update due to invalid gradients.")
            return self.prev_dynamic_lambda

        dynamic_lambda = self._alignment_score_stable_ref(grads_ct_flat, grads_reg_flat).item()
        
        # EMA smoothing on lambda
        dynamic_lambda = self.ema_scale * dynamic_lambda + (1 - self.ema_scale) * self.prev_dynamic_lambda
        self.prev_dynamic_lambda = dynamic_lambda

        return dynamic_lambda
