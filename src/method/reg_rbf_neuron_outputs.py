import torch

from src.method.method_plugin_abc import MethodPluginABC
from util.tensor import pad_zero_dim0


class RBFNeuronOutReg(MethodPluginABC):
    """
    RBF neuron output regularization to overcome catastrophic forgetting problem.

    Attributes:
        params_buffer (dict): A buffer to store the parameters of the model.

    Methods:
        setup_task(task_id: int):
            Placeholder method for setting up a task. Currently does nothing.

        forward(x, y, loss, preds):
            Adjusts the loss by combining the original loss with a RBF neuron 
            output regularization loss.
    """

    def __init__(self, 
        alpha: float,
        eps: float = 1e-6
    ):
        """
        Initializes the RBF neuron output regularization method within the whole domain
        with the given parameters.

        Args:
            alpha (float): 
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.
        """
                
        super().__init__()

        self.params_buffer = {}
        self.importance = {}

        self.alpha = alpha


    def setup_task(self, task_id: int):
        """
        Sets up a task with the given task ID.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """


    def forward(self, x, y, loss, preds):
        """
        Perform the forward pass of the RBF neuron output regularization method.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Initial loss value.
            preds (torch.Tensor): Predictions tensor.
            
        Returns:
            tuple: Updated loss and predictions tensors.
        """

    def gaussian_integral(self, c1, sigma1, c2, sigma2):
        """
        Computes the Gaussian integral:
        ∫ exp( -|| (x - c1) / sigma1 ||² - || (x - c2) / sigma2 ||² ) dx

        Args:
        - c1, c2: Tensors of shape (K, d) representing centers of Gaussians.
        - sigma1, sigma2: Tensors of shape (K, d) representing standard deviations.

        Returns:
        - Tensor of shape (K,) with computed integral values.
        """

        d = c1.shape[1]
        A = torch.diag_embed(1 / sigma1**2)  # (K, d, d)
        B = torch.diag_embed(1 / sigma2**2)  # (K, d, d)

        A_B = A + B  # (K, d, d)
        A_B_inv = torch.linalg.inv(A_B)  # (K, d, d)

        det_A = torch.prod(sigma1**-2, dim=1)  # Determinant of A (K,)
        det_B = torch.prod(sigma2**-2, dim=1)  # Determinant of B (K,)
        det_A_B = torch.prod(sigma1**-2 + sigma2**-2, dim=1)  # Determinant of (A+B) (K,)

        exp_term = torch.einsum('bi,bij,bj->b', c1 - c2, A_B_inv, c1 - c2)  # Quadratic form

        integral = (
            (torch.pi ** (d / 2)) 
            * torch.sqrt(det_A) 
            * torch.sqrt(det_B) 
            / torch.sqrt(det_A_B)
            * torch.exp(-exp_term)
        )

        return integral  # Shape (K,)


    def compute_integral_gaussian(self, W, delta_W, C, delta_C, Sigma, delta_Sigma):
        """
        Computes the full integral based on Gaussian kernel parametrization.

        Args:
        - W: Tensor of shape (K, M) containing weights w_j.
        - delta_W: Tensor of shape (K, M) containing perturbations Δw_j.
        - C: Tensor of shape (K, d) representing Gaussian centers c_j.
        - delta_C: Tensor of shape (K, d) representing perturbations Δc_j.
        - Sigma: Tensor of shape (K, d) representing standard deviations σ_j.
        - delta_Sigma: Tensor of shape (K, d) representing perturbations Δσ_j.

        Returns:
        - The computed integral values as a tensor of shape (M,).
        """

        # Compute integrals ∫ e_j(x) e_i(x) dx
        E_integrals = self.gaussian_integral(C + delta_C, Sigma + delta_Sigma, C + delta_C, Sigma + delta_Sigma)  # (K,)

        # Compute integrals ∫ f_j(x) f_i(x) dx
        F_integrals = self.gaussian_integral(C, Sigma, C, Sigma)  # (K,)

        # Compute integrals ∫ e_j(x) f_i(x) dx
        EF_integrals = self.gaussian_integral(C + delta_C, Sigma + delta_Sigma, C, Sigma)  # (K,)

        # Expand tensors for element-wise broadcasting
        W1 = W.unsqueeze(1)  # Shape (K, 1, M)
        W2 = W.unsqueeze(0)  # Shape (1, K, M)
        Delta_W1 = delta_W.unsqueeze(1)  # Shape (K, 1, M)
        Delta_W2 = delta_W.unsqueeze(0)  # Shape (1, K, M)

        # Compute the three terms in the expression
        first_term = torch.sum((W1 + Delta_W1) * (W2 + Delta_W2) * E_integrals.unsqueeze(-1), dim=(0, 1))
        second_term = -2 * torch.sum((W1 + Delta_W1) * W2 * EF_integrals.unsqueeze(-1), dim=(0, 1))
        third_term = torch.sum(W1 * W2 * F_integrals.unsqueeze(-1), dim=(0, 1))

        # Final integral value
        integral_value = first_term + second_term + third_term  # Shape (M,)

        return integral_value

