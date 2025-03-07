import logging
import torch
import torch.nn.functional as F
from src.method.method_plugin_abc import MethodPluginABC

from copy import deepcopy

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Dreaming(MethodPluginABC):
    """
    This class synthesizes new samples based on the last Gaussian classification layer.
    
    Attributes:
        params_buffer (dict): Stores model parameters for regularization.
        alpha (float): Regularization strength.
        dreamed_data_buffer (dict): Stores dreamed data for regularization of the penultimate layer.
        old_nclasses (int): Indices of all classes seen up to the current task (exclusively)
    """

    def __init__(self, alpha: float):
        """
        Initializes the dreaming method.

        Args:
            alpha (float): Regularization coefficient.
        """
        super().__init__()
        self.params_buffer = {}
        self.dreamed_data_buffer = {}
        self.old_nclasses = None

        self.alpha = alpha
        log.info(f"Dreaming initialized with alpha={alpha}")

    def setup_task(self, task_id: int):
        """
        Sets up a new task and stores parameters from the incremental classification head (RBFHead).

        Args:
            task_id (int): Unique task identifier.
        """

        self.params_buffer = {}
        self.dreamed_data_buffer = {}
        
        def extract_RBFHead_params(layer, idx, classes):
            if type(layer).__name__ == "RBFHeadLayer":
                kernels_centers = layer.kernels_centers[:classes, :].detach().clone()
                log_shapes = layer.log_shapes[:classes, :].detach().clone()
                return {
                    f"kernels_centers_{idx}": kernels_centers,
                    f"log_shapes_{idx}": log_shapes
                }
            
        self.task_id = task_id

        if task_id > 0:
            model = self.module.module
            for module in model.children():
                if type(module).__name__ == "IncrementalClassifier":
                    layer = module.classifier
                    self.old_nclasses = module.old_nclasses
                    if type(layer).__name__ == "RBFHeadLayer":
                        self.params_buffer.update(extract_RBFHead_params(layer, "head",self. old_nclasses))

            with torch.no_grad():   
                self.old_module = deepcopy(self.module)
                for p in self.old_module.parameters():
                    p.requires_grad = False
                self.old_module.eval()

    def generate_dreamed_data(self, no_samples_per_class: int, steps: int = 500, lr: float = 0.1, 
                          lambda_l2: float = 0.01, 
                          lambda_tv: float = 0.001):
        """
        Generate dreamed images such that their activations match stored Gaussian statistics 
        in the last classification layer while also ensuring realism with L2 and TV losses.

        Args:
            no_samples_per_class (int): Number of synthesized samples per class.
            steps (int): Number of optimization steps.
            lr (float): Learning rate for optimization.
            lambda_l2 (float): Weight for L2 prior loss.
            lambda_tv (float): Weight for total variation loss.
        """

        self.old_module.eval()
        kernels_centers = self.params_buffer["kernels_centers_head"]
        sigmas = self.params_buffer["log_shapes_head"].exp()

        self.dreamed_data_buffer = {}

        in_shape = kernels_centers.shape[0]

        for class_idx, (mean, cov) in enumerate(zip(kernels_centers, sigmas)):
            mean = mean.to(dtype=torch.float32)
            cov = cov.to(dtype=torch.float32)

            batch_size = no_samples_per_class
            dreamed_images = torch.randn((batch_size, *in_shape), requires_grad=True)

            optimizer = torch.optim.Adam([dreamed_images], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()

                _ = self.old_module(dreamed_images)
                stored_activations = self.old_module.activations[-1]

                act_mean = stored_activations.mean(dim=0)
                act_var = stored_activations.var(dim=0, unbiased=False)

                # Loss: Match activation mean & variance
                mean_loss = F.mse_loss(act_mean, mean)
                cov_loss = F.mse_loss(act_var, cov)

                # L2 prior loss (keeps images from drifting too far)
                l2_loss = torch.norm(dreamed_images, p=2) / batch_size

                # Total variation loss (removes noise & artifacts)
                tv_loss = torch.mean(torch.abs(dreamed_images[:, :, :-1, :] - dreamed_images[:, :, 1:, :])) + \
                        torch.mean(torch.abs(dreamed_images[:, :, :, :-1] - dreamed_images[:, :, :, 1:]))

                # Total loss: Activation match + Regularization
                total_loss = mean_loss + cov_loss + lambda_l2 * l2_loss + lambda_tv * tv_loss

                # Backpropagate
                total_loss.backward()
                optimizer.step()

            # Store the dreamed data in buffer
            dreamed_images = dreamed_images.detach()

            self.dreamed_data_buffer[class_idx] = dreamed_images
    

    def forward(self, x, y, loss, preds):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss (torch.Tensor): Initial loss.
            preds (torch.Tensor): Model predictions.
        
        Returns:
            tuple: Updated loss and predictions.
        """
        if self.task_id > 0:
            penultimate_activations_current = self.module.activations[-1]
            _ = self.old_module(x)
            penultimate_activations_old = self.old_module.activations[-1].clone().detach()

            kernels_centers = self.params_buffer["kernels_centers_head"]
            sigma = self.params_buffer["log_shapes_head"].exp()

            batch_size = x.size(0)
            out_features, in_features = kernels_centers.shape
           
            c = kernels_centers.expand(batch_size, out_features, in_features)
            sigma = sigma.expand(batch_size, out_features, in_features)
            penultimate_activations_old = penultimate_activations_old.view(batch_size, 1, in_features)
            penultimate_activations_current = penultimate_activations_current.view(batch_size, 1, in_features)

            weight_factor = torch.exp(-((penultimate_activations_old - c) / sigma) ** 2)
            weighted_diff = weight_factor * (penultimate_activations_old - penultimate_activations_current)
            reg_loss = self.alpha * torch.norm(weighted_diff, p=2)
            loss += reg_loss
        
        return loss, preds