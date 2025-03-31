import logging
import torch
from torch.nn import functional as F


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class DynamicScaling():

    def __init__(self, module, min_lambda=0.1, max_lambda=2.0, beta=1e-5):
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
        self.dynamic_lambda = 1.0
        self.module = module

    def forward(self, task_id, loss_ce, loss_reg):
        """Compute the dynamically scaled loss based on gradient alignment."""
        
        if task_id > 0 and self.module.training:

            grads_reg = torch.autograd.grad(loss_reg, self.module.parameters(), retain_graph=True)            
            grads_ce = torch.autograd.grad(loss_ce, self.module.parameters(), retain_graph=True)

            dynamic_lambda = self.compute_lambda_t(grads_ce, grads_reg)
            self.dynamic_lambda = dynamic_lambda
        return self.dynamic_lambda * loss_ce + loss_reg

    def compute_lambda_t(self, grads_ce, grads_reg):
        """Compute lambda_t using exponential annealing of misaligned gradients."""
        # Flatten gradients and remove None values
        grads_ce_flat = torch.cat([g.flatten() for g in grads_ce if g is not None], dim=0)
        grads_reg_flat = torch.cat([g.flatten() for g in grads_reg if g is not None], dim=0)

        # Prevent cosine similarity errors if gradients are zero
        if grads_ce_flat.numel() == 0 or grads_reg_flat.numel() == 0:
            log.warning("Skipping lambda_t update due to missing gradients.")
            return self.dynamic_lambda  # Keep previous value

        # Compute cosine similarity safely
        cos_theta = F.cosine_similarity(
            grads_ce_flat.unsqueeze(0), grads_reg_flat.unsqueeze(0), dim=1, eps=1e-8
        ).item()

        # Ensure we do not scale in opposite directions
        if cos_theta < 0:
            dynamic_lambda = self.min_lambda  # Minimum scaling to prevent opposition
        else:
            dynamic_lambda = torch.exp(torch.tensor(-self.beta * (1 - cos_theta))).item()

        # Ensure lambda_t stays within limits
        return max(self.min_lambda, min(dynamic_lambda, self.max_lambda))
