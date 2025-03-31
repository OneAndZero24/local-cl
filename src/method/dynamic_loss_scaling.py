import logging
import torch
from torch.nn import functional as F


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DynamicScaling():

    def __init__(self, module, min_lambda, max_lambda, beta, threshold, ema_scale):
        """
        Continual learning model using dynamic lambda_t with entropy & momentum.
        Args:
            min_lambda: Minimum lambda value for scaling CE loss.
            max_lambda: Maximum lambda value for scaling CE loss.
            beta: Momentum factor for smoothing lambda_t updates.
        """
        super().__init__()
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.beta = beta
        self.threshold = threshold
        self.ema_scale = ema_scale
        self.module = module

        self.prev_dynamic_lambda = None

    def forward(self, task_id, loss_ce, loss_reg):
        """Compute the dynamically scaled loss based on gradient alignment."""
        
        if task_id > 0 and self.module.training:
            grads_reg = torch.autograd.grad(loss_reg, self.module.parameters(), retain_graph=True)            
            grads_ce = torch.autograd.grad(loss_ce, self.module.parameters(), retain_graph=True)
            dynamic_lambda = self.compute_lambda_t(grads_ce, grads_reg)
        else:
            dynamic_lambda = 1.0
        return dynamic_lambda * loss_ce + loss_reg

    def compute_lambda_t(self, grads_ce, grads_reg):
        """Compute lambda_t using exponential annealing of misaligned gradients."""

        grads_ce_flat = torch.cat([g.flatten() for g in grads_ce if g is not None], dim=0)
        grads_reg_flat = torch.cat([g.flatten() for g in grads_reg if g is not None], dim=0)

        if grads_ce_flat.numel() == 0 or grads_reg_flat.numel() == 0:
            log.warning("Skipping lambda_t update due to missing gradients.")
            return self.prev_dynamic_lambda 

        # Compute cosine similarity safely
        cos_theta = F.cosine_similarity(
            grads_ce_flat.unsqueeze(0), 
            grads_reg_flat.unsqueeze(0), dim=1, eps=1e-8
        ).item()

        if cos_theta < self.threshold:
            dynamic_lambda = self.min_lambda
        else:
            dynamic_lambda = torch.exp(torch.tensor(-self.beta * (1 - cos_theta))).item()

        if self.prev_dynamic_lambda is None:
            self.prev_dynamic_lambda = self.min_lambda
        else:
            dynamic_lambda = self.ema_scale * dynamic_lambda + (1 - self.ema_scale) * self.prev_dynamic_lambda
            self.prev_dynamic_lambda = dynamic_lambda

        return max(self.min_lambda, min(dynamic_lambda, self.max_lambda))
