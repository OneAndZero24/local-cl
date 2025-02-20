import torch

from src.method.method_plugin_abc import MethodPluginABC

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

    def compute_log_gaussian_convolution(self, c1, sigma1, c2, sigma2):
        """
        Computes the log of Gaussian convolution integral:
        log(∫ exp( - (x - c1)^T A^{-1} (x - c1) - (x - c2)^T B^{-1} (x - c2) ) dx)
        """

        # Dimension of x
        d = c1.shape[1]

        c_diff = c1 - c2

        corr_mat_1 = torch.diag_embed(sigma1**2)
        corr_mat_2 = torch.diag_embed(sigma2**2)
        corr_mat_12 = corr_mat_1 + corr_mat_2

        corr_mat_inv_12 = torch.linalg.inv(corr_mat_12)

        # Compute log-determinants, shape (K,)
        log_det_corr_mat_1 = torch.logdet(corr_mat_1)
        log_det_corr_mat_2 = torch.logdet(corr_mat_2)
        log_det_corr_mat_12 = torch.logdet(corr_mat_12)

        # Quadratic exponent term        
        exp_term = torch.einsum('kij,kj->ki', corr_mat_inv_12, c_diff)  # Shape (K, d)
        exp_term = torch.einsum('ki,ji->kj', c_diff, exp_term)  # Shape (K, K), symmetric
        
        # Compute final integral in log-space first
        log_integral = (
            torch.tensor(d / 2) * torch.log(torch.tensor(torch.pi))
            + 0.5 * (log_det_corr_mat_1 + log_det_corr_mat_2 - log_det_corr_mat_12)
            - exp_term
        )
    
        return log_integral

    def compute_integral_gaussian(self, W_old, W_curr, C_old, C_curr, Sigma_old, Sigma_curr):
        """
        Computes the logarithm of the integral of the square of the difference between the old neuron's response and the current one.

        Args:
            W_old (torch.Tensor): Tensor of shape (K, M) containing the old weights w_j.
            W_curr (torch.Tensor): Tensor of shape (K, M) containing the current weights w_j.
            C_old (torch.Tensor): Tensor of shape (K, d) representing the old Gaussian centers c_j.
            C_curr (torch.Tensor): Tensor of shape (K, d) representing the current Gaussian centers c_j.
            Sigma_old (torch.Tensor): Tensor of shape (K, d) representing the old standard deviations σ_j.
            Sigma_curr (torch.Tensor): Tensor of shape (K, d) representing the current standard deviations σ_j.

        Returns:
            torch.Tensor: A tensor of shape (M,) containing the computed logarithmic integral values.
        """

        # Compute integrals log(∫ e_j(x) e_i(x) dx)
        E_integrals = self.compute_log_gaussian_convolution(C_curr, Sigma_curr, C_curr, Sigma_curr)  # (K,K)
        E_integrals_max = E_integrals.max()

        # Compute integrals log(∫ f_j(x) f_i(x) dx)
        F_integrals = self.compute_log_gaussian_convolution(C_old, Sigma_old, C_old, Sigma_old)  # (K,K)
        F_integrals_max = F_integrals.max()

        # Compute integrals log(∫ e_j(x) f_i(x) dx)
        EF_integrals = self.compute_log_gaussian_convolution(C_curr, Sigma_curr, C_old, Sigma_old)  # (K,K)
        EF_integrals_max = EF_integrals.max()

        integrals_max = torch.stack([
            E_integrals_max,
            EF_integrals_max,
            F_integrals_max
        ]).max()

        EF_integrals = torch.exp(EF_integrals - integrals_max)
        F_integrals = torch.exp(F_integrals - integrals_max)
        E_integrals = torch.exp(E_integrals - integrals_max)

        first_term = W_curr.T @ E_integrals @ W_curr
        second_term = W_curr.T @ EF_integrals @ W_old
        third_term = W_old.T @ F_integrals @ W_old

        exp_term = torch.log((first_term + second_term + third_term).relu() + self.eps)
        log_integral_value = (integrals_max + exp_term).mean()

        return log_integral_value