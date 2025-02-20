import torch

from src.method.method_plugin_abc import MethodPluginABC
from util.tensor import pad_zero_dim0

from copy import deepcopy

class RBFNeuronOutReg(MethodPluginABC):
    """
    RBF neuron output regularization to mitigate catastrophic forgetting.
    
    Attributes:
        params_buffer (dict): Stores model parameters for regularization.
        alpha (float): Regularization strength.
        eps (float): Small value to prevent division by zero.
    """

    def __init__(self, alpha: float, eps: float = 1e-6):
        """
        Initializes the RBF neuron output regularization method.

        Args:
            alpha (float): Regularization coefficient.
            eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.params_buffer = {}
        self.importance = {}
        self.alpha = alpha
        self.eps = eps

    def setup_task(self, task_id: int):
        """
        Sets up a new task and stores parameters from RBF layers.

        Args:
            task_id (int): Unique task identifier.
        """
        self.task_id = task_id

        if task_id > 0:
            model = self.module.module  # Access the wrapped model
            for module in model.children():
                if isinstance(module, torch.nn.ModuleList):
                    for idx, layer in enumerate(module):
                        if type(layer).__name__ == "RBFLayer":
                            self.params_buffer.update({
                                f"weights_{idx}": layer.weights.detach().clone(),
                                f"kernels_centers_{idx}": layer.kernels_centers.detach().clone(),
                                f"log_shapes_{idx}": layer.log_shapes.detach().clone(),
                            })

    def forward(self, x, y, loss, preds):
        """
        Forward pass with RBF neuron output regularization.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Initial loss.
            preds (torch.Tensor): Model predictions.
        
        Returns:
            tuple: Updated loss and predictions.
        """
        if self.task_id > 0:
            for module in self.module.module.children():
                if isinstance(module, torch.nn.ModuleList):
                    for idx, layer in enumerate(module):
                        if type(layer).__name__ == "RBFLayer":
                            # Get current parameters
                            W_curr = layer.weights.T
                            C_curr = layer.kernels_centers
                            Sigma_curr = layer.log_shapes.exp()

                            # Retrieve old stored parameters
                            W_old = self.params_buffer[f"weights_{idx}"].T
                            C_old = self.params_buffer[f"kernels_centers_{idx}"]
                            Sigma_old = self.params_buffer[f"log_shapes_{idx}"].exp()

                            # Add regularization loss
                            loss += self.alpha * self.compute_integral_gaussian(
                                W_old=W_old, W_curr=W_curr,
                                C_old=C_old, C_curr=C_curr,
                                Sigma_old=Sigma_old, Sigma_curr=Sigma_curr
                            )

        return loss, preds

    def compute_gaussian_convolution(self, c1, sigma1, c2, sigma2):
        """
        Computes the Gaussian integral:
        log(∫ exp( - (x - c1)^T A^{-1} (x - c1) - (x - c2)^T B^{-1} (x - c2) ) dx)
        """

        d = c1.shape[1]  # Dimension of x

        # Covariance matrices (diagonal)
        A = torch.diag_embed(sigma1**2)  # (K, d, d)
        B = torch.diag_embed(sigma2**2)  # (K, d, d)

        # Compute A + B
        A_B = A + B + self.eps * torch.eye(d, device=A.device).expand(A.shape)  # (K, d, d) for numerical stability
        A_B_inv = torch.linalg.inv(A_B)  # Inverse of (A + B)

        # Compute log-determinants
        log_det_A = torch.sum(torch.log(sigma1**2), dim=1)  # log(det(A)) (K,)
        log_det_B = torch.sum(torch.log(sigma2**2), dim=1)  # log(det(B)) (K,)
        log_det_A_B = torch.sum(torch.log(sigma1**2 + sigma2**2), dim=1)  # log(det(A+B)) (K,)

        # Quadratic exponent term
        exp_term = torch.einsum('bi,bij,bj->b', c1 - c2, A_B_inv, c1 - c2)  # (K,)

        # Compute final integral in log-space first
        log_integral = (
            torch.tensor(d / 2) * torch.log(torch.tensor(torch.pi))
            + 0.5 * (log_det_A + log_det_B - log_det_A_B)
            - exp_term
        )

        return log_integral.exp()

    def compute_integral_gaussian(self, W_old, W_curr, C_old, C_curr, Sigma_old, Sigma_curr):
        """
        Computes the full integral based on Gaussian kernel parametrization.

        Args:
        - W_old: Tensor of shape (K, M) containing weights w_j.
        - W_curr: Tensor of shape (K, M) containing perturbations Δw_j.
        - C_old: Tensor of shape (K, d) representing Gaussian centers c_j.
        - C_curr: Tensor of shape (K, d) representing perturbations Δc_j.
        - Sigma_old: Tensor of shape (K, d) representing standard deviations σ_j.
        - Sigma_curr: Tensor of shape (K, d) representing perturbations Δσ_j.

        Returns:
        - The computed integral values as a tensor of shape (M,).
        """

        # Compute integrals ∫ e_j(x) e_i(x) dx
        E_integrals = self.compute_gaussian_convolution(C_curr, Sigma_curr, C_curr, Sigma_curr)  # (K,)

        # Compute integrals ∫ f_j(x) f_i(x) dx
        F_integrals = self.compute_gaussian_convolution(C_old, Sigma_old, C_old, Sigma_old)  # (K,)

        # Compute integrals ∫ e_j(x) f_i(x) dx
        EF_integrals = self.compute_gaussian_convolution(C_curr, Sigma_curr, C_old, Sigma_old)  # (K,)

        # Expand tensors for element-wise broadcasting
        W1_old = W_old.unsqueeze(1)  # Shape (K, 1, M)
        W2_old = W_old.unsqueeze(0)  # Shape (1, K, M)

        W1_curr = W_curr.unsqueeze(1)  # Shape (K, 1, M)
        W2_curr = W_curr.unsqueeze(0)  # Shape (1, K, M)

        # Compute the three terms in the expression
        first_term = torch.sum(W1_curr * W2_curr * E_integrals.unsqueeze(-1), dim=(0, 1))
        second_term = -2 * torch.sum(W1_curr * W2_old * EF_integrals.unsqueeze(-1), dim=(0, 1))
        third_term = torch.sum(W1_old * W2_old * F_integrals.unsqueeze(-1), dim=(0, 1))

        # Final integral value
        integral_value = first_term + second_term + third_term  # Shape (M,)
        print(integral_value.mean())
        return integral_value.mean()