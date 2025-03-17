import logging

import torch

from src.method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class RBFNeuronOutReg(MethodPluginABC):
    """
    RBF neuron output regularization to mitigate catastrophic forgetting.
    
    Attributes:
        params_buffer (dict): Stores model parameters for regularization.
        alpha (float): Regularization strength.
        eps (float): Safety net value.
    """


    def __init__(self, alpha: float):
        """
        Initializes the RBF neuron output regularization method.

        Args:
            alpha (float): Regularization coefficient.
        """

        super().__init__()
        self.params_buffer = {}
        self.alpha = alpha
        log.info(f"Initialized RBFNeuronOutReg with alpha={alpha}")


    def setup_task(self, task_id: int):
        """
        Sets up a new task and stores parameters from RBF layers.

        Args:
            task_id (int): Unique task identifier.
        """

        def get_rbf_layer_params(layer, idx, classes=None, head_mode=None):
            if classes is not None:
                assert head_mode in ["single", "multi"], "Please provide correct head mode."

                if head_mode == "single":
                    weights = None
                    kernels_centers = layer.kernels_centers[:classes, :].detach().clone()
                    log_shapes = layer.log_shapes[:classes, :].detach().clone()
                elif head_mode == "multi":
                    weights = layer.weights[:classes, :].detach().clone()
                    kernels_centers = layer.kernels_centers.detach().clone()
                    log_shapes = layer.log_shapes.detach().clone()
            else:
                weights = layer.weights.detach().clone()
                kernels_centers = layer.kernels_centers.detach().clone()
                log_shapes = layer.log_shapes.detach().clone()
            return {
                f"weights_{idx}": weights,
                f"kernels_centers_{idx}": kernels_centers,
                f"log_shapes_{idx}": log_shapes
            }

        self.task_id = task_id

        if task_id > 0:
            model = self.module.module
            for module in model.children():
                if isinstance(module, torch.nn.ModuleList):
                    for idx, layer in enumerate(module):
                        if type(layer).__name__ == "RBFLayer":
                            self.params_buffer.update(get_rbf_layer_params(layer, idx))
                elif type(module).__name__ == "IncrementalClassifier":
                    layer = module.classifier
                    old_nclasses = module.old_nclasses
                    if type(layer).__name__ == "SingleRBFHeadLayer":
                        self.params_buffer.update(get_rbf_layer_params(layer, "head", old_nclasses, "single"))
                    elif type(layer).__name__ == "RBFLayer":
                        self.params_buffer.update(get_rbf_layer_params(layer, "head", old_nclasses, "multi"))


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

        def get_rbf_layer_params(layer, classes=None, head_mode=None):
            if classes is not None:
                assert head_mode in ["single", "multi"], "Please provide correct head mode"
                if head_mode == "single":
                    W_curr = None
                    C_curr = layer.kernels_centers[:classes,:]
                    Sigma_curr = layer.log_shapes[:classes,:].exp()
                elif head_mode == "multi":
                    W_curr = layer.weights[:classes,:].T
                    C_curr = layer.kernels_centers
                    Sigma_curr = layer.log_shapes.exp()
            else:
                W_curr = layer.weights.T
                C_curr = layer.kernels_centers
                Sigma_curr = layer.log_shapes.exp()
            return W_curr, C_curr, Sigma_curr
        
        def get_old_rbf_layer_params(idx):
            W_old = self.params_buffer.get(f"weights_{idx}", None)
            W_old = W_old.T if W_old is not None else W_old
            C_old = self.params_buffer[f"kernels_centers_{idx}"]
            Sigma_old = self.params_buffer[f"log_shapes_{idx}"].exp()
            return W_old, C_old, Sigma_old

        if self.task_id > 0:
            for module in self.module.module.children():
                if isinstance(module, torch.nn.ModuleList):
                    for idx, layer in enumerate(module):
                        if type(layer).__name__ == "RBFLayer":
                            # Get current parameters
                            W_curr, C_curr, Sigma_curr = get_rbf_layer_params(layer)

                            # Retrieve old stored parameters
                            W_old, C_old, Sigma_old = get_old_rbf_layer_params(idx)
    
                            # Add regularization loss
                            loss += self.alpha * self.compute_hidden_integral_gaussian(
                                W_old=W_old, W_curr=W_curr,
                                C_old=C_old, C_curr=C_curr,
                                Sigma_old=Sigma_old, Sigma_curr=Sigma_curr
                            )
                elif type(module).__name__ == "IncrementalClassifier":
                    layer = module.classifier
                    old_nclasses = module.old_nclasses
                    if type(layer).__name__ == "SingleRBFHeadLayer":
                        # Get current parameters
                        _, C_curr, Sigma_curr = get_rbf_layer_params(layer, old_nclasses, "single")

                        # Retrieve old stored parameters
                        _, C_old, Sigma_old = get_old_rbf_layer_params("head")
                        loss += self.alpha * self.compute_head_integral_gaussian(
                            C_old=C_old, C_curr=C_curr,
                            Sigma_old=Sigma_old, Sigma_curr=Sigma_curr
                        )

                    elif type(layer).__name__ == "RBFLayer":
                        # Get current parameters
                        W_curr, C_curr, Sigma_curr = get_rbf_layer_params(layer, old_nclasses, "multi")

                        # Retrieve old stored parameters
                        W_old, C_old, Sigma_old = get_old_rbf_layer_params("head")

                        # Add regularization loss
                        loss += self.alpha * self.compute_hidden_integral_gaussian(
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


    def compute_hidden_integral_gaussian(self, W_old, W_curr, C_old, C_curr, Sigma_old, Sigma_curr):
        """
        Computes the integral of the square of the difference between the old neuron's response and the current one
        for a hidden RBF layer.

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

        # Ensure numerical stability
        final_integral = torch.clamp(final_integral, min=0)

        # torch.exp(F_integrals_max) is a constant and has to be included to
        # prevent overflow. It does not change the final minimum.
        final_integral = torch.exp(integrals_max - F_integrals_max) * final_integral

        final_integral = final_integral.mean()

        return final_integral
    
    def compute_head_integral_gaussian(self, C_old, C_curr, Sigma_old, Sigma_curr):
        """
        Computes the integral of the squared difference between the old and current neuron responses
        for an RBF layer used as an incremental classification head.

        Args:
            C_old (torch.Tensor): Tensor of shape (K, d) representing old Gaussian centers.
            C_curr (torch.Tensor): Tensor of shape (K, d) representing current Gaussian centers.
            Sigma_old (torch.Tensor): Tensor of shape (K, d) representing old standard deviations.
            Sigma_curr (torch.Tensor): Tensor of shape (K, d) representing current standard deviations.

        Returns:
            torch.Tensor: A tensor of shape (K,) containing the computed integral values.
        """
        d = C_old.shape[1]  # Dimensionality

        # Compute squared integral of each Gaussian
        integral_old = (torch.pi ** (d / 2)) * torch.prod(Sigma_old / torch.sqrt(torch.tensor(2.0, device=Sigma_old.device)), dim=1)
        integral_curr = (torch.pi ** (d / 2)) * torch.prod(Sigma_curr / torch.sqrt(torch.tensor(2.0, device=Sigma_curr.device)), dim=1)

        # Compute convolution integral
        combined_sigma = torch.sqrt(Sigma_old**2 + Sigma_curr**2)
        integral_cross = (torch.pi ** (d / 2)) * torch.prod(Sigma_old * Sigma_curr / combined_sigma, dim=1) * \
                        torch.exp(-torch.sum((C_old - C_curr) ** 2 / (Sigma_old ** 2 + Sigma_curr ** 2), dim=1))

        # Compute final integral value
        integral_value = integral_curr + integral_old - 2 * integral_cross

        # Ensure numerical stability
        integral_value = torch.clamp(integral_value, min=0)

        return integral_value.mean()

        

        