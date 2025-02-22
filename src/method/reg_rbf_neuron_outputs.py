import torch

from src.method.method_plugin_abc import MethodPluginABC
from src.model import IncrementalClassifier, LayerType
from src.model.layer.rbf import RBFLayer
class RBFNeuronOutReg(MethodPluginABC):
    """
    RBF neuron output regularization to mitigate catastrophic forgetting.
    
    Attributes:
        params_buffer (dict): Stores model parameters for regularization.
        alpha (float): Regularization strength.
        eps (float): Safety net value.
    """

    def __init__(self, alpha: float, eps: float = 1e-6):
        """
        Initializes the RBF neuron output regularization method.

        Args:
            alpha (float): Regularization coefficient.
            eps (float, optional): Safety net value. Defaults to 1e-6.
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
            model = self.module.module
            for module in model.children():
                if isinstance(module, torch.nn.ModuleList):
                    for idx, layer in enumerate(module):
                        if type(layer).__name__ == "RBFLayer":
                            self.params_buffer.update({
                                f"weights_{idx}": layer.weights.detach().clone(),
                                f"kernels_centers_{idx}": layer.kernels_centers.detach().clone(),
                                f"log_shapes_{idx}": layer.log_shapes.detach().clone(),
                            })
                elif type(module).__name__ == "IncrementalClassifier":
                    layer = module.classifier
                    old_nclasses = module.old_nclasses
                    if type(layer).__name__ == "RBFLayer":
                        self.params_buffer.update({
                                f"weights_head": layer.weights[:old_nclasses,:].detach().clone(),
                                f"kernels_centers_head": layer.kernels_centers.detach().clone(),
                                f"log_shapes_head": layer.log_shapes.detach().clone(),
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
                elif type(module).__name__ == "IncrementalClassifier":
                    layer = module.classifier
                    old_nclasses = module.old_nclasses
                    if type(layer).__name__ == "RBFLayer":
                         # Get current parameters
                            W_curr = layer.weights[:old_nclasses,:].T
                            C_curr = layer.kernels_centers
                            Sigma_curr = layer.log_shapes.exp()

                            # Retrieve old stored parameters
                            W_old = self.params_buffer[f"weights_head"].T
                            C_old = self.params_buffer[f"kernels_centers_head"]
                            Sigma_old = self.params_buffer[f"log_shapes_head"].exp()

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

        # Compute squared sigma to form diagonal covariance matrices
        sigma1_sq = sigma1**2  # (K, d)
        sigma2_sq = sigma2**2  # (K, d)
        corr_mat_12 = sigma1_sq[:, None, :] + sigma2_sq[None, :, :]  # (K, K, d)

        # Compute log-determinants
        log_det_corr_mat_1 = torch.sum(torch.log(sigma1_sq), dim=1, keepdim=True)  # (K, 1)
        log_det_corr_mat_2 = torch.sum(torch.log(sigma2_sq), dim=1, keepdim=True)  # (K, 1)
        log_det_corr_mat_12 = torch.sum(torch.log(corr_mat_12), dim=2)  # (K, K)

        # Quadratic exponent term (c1 - c2)^T * (corr_mat_12)^(-1) * (c1 - c2)
        c_diff = c1[:, None, :] - c2[None, :, :]  # (K, K, d)

        # This trick sppeds up computations
        exp_term = torch.sum((c_diff**2) / corr_mat_12, dim=2)  # (K, K)

        # Compute final integral in log-space
        d = c1.shape[1]
        log_integral = (
            (d / 2) * torch.log(torch.tensor(torch.pi))
            + 0.5 * (log_det_corr_mat_1 + log_det_corr_mat_2.T - log_det_corr_mat_12) # Determinants are broadcasted here
            - exp_term
        )

        return log_integral  # (K, K)

    def compute_integral_gaussian(self, W_old, W_curr, C_old, C_curr, Sigma_old, Sigma_curr):
        """
        Computes the integral of the square of the difference between the old neuron's response and the current one.

        Args:
            W_old (torch.Tensor): Tensor of shape (K, M) containing the old weights w_j.
            W_curr (torch.Tensor): Tensor of shape (K, M) containing the current weights w_j.
            C_old (torch.Tensor): Tensor of shape (K, d) representing the old Gaussian centers c_j.
            C_curr (torch.Tensor): Tensor of shape (K, d) representing the current Gaussian centers c_j.
            Sigma_old (torch.Tensor): Tensor of shape (K, d) representing the old standard deviations σ_j.
            Sigma_curr (torch.Tensor): Tensor of shape (K, d) representing the current standard deviations σ_j.

        Returns:
            torch.Tensor: A tensor of shape (M,) containing the computed integral values.
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
        
        first_term = torch.sum(W_curr * (E_integrals @ W_curr), dim=0)
        second_term = (-2)*torch.sum(W_curr * (EF_integrals @ W_old), dim=0)
        third_term = torch.sum(W_old * (F_integrals @ W_old), dim=0)
        
        final_integral = first_term + second_term + third_term

        assert (final_integral + self.eps).all() >= 0, f"This expression has to be greater than 0"

        # torch.exp(F_integrals_max) is a constant and has to be included to
        # prevent overflow. It does not change the final minimum.
        final_integral = torch.exp(integrals_max - F_integrals_max) * final_integral
        final_integral = final_integral.mean()

        return final_integral