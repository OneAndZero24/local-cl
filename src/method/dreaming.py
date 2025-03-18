import logging
import torch
import torch.nn.functional as F
from src.method.method_plugin_abc import MethodPluginABC
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Tuple
import ast

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

    def __init__(self, alpha: float, no_samples_per_class: int = 32, steps: int = 500,
                 dreaming_lr: float = 0.1, lambda_l2: float = 0.01, lambda_tv: float = 0.001,
                 in_shape: Tuple[int] = (1,32,32), diversity_weight: float = 0.05):
        """
        Initializes the dreaming method.

        Args:
            alpha (float): Regularization coefficient.
        """
        super().__init__()
        self.params_buffer = {}
        self.dreamed_data_buffer = {}
        self.old_nclasses = None

        self.no_samples_per_class = no_samples_per_class
        self.steps = steps
        self.dreaming_lr = dreaming_lr
        self.lambda_l2 = lambda_l2
        self.lambda_tv = lambda_tv
        self.diversity_weight = diversity_weight
        self.in_shape = ast.literal_eval(in_shape)

        self.alpha = alpha
        log.info(f"Dreaming initialized with alpha={alpha}")

    def setup_task(self, task_id: int):
        """
        Sets up a new task and stores parameters from the incremental classification head.

        Args:
            task_id (int): Unique task identifier.
        """

        self.params_buffer = {}
        self.dreamed_data_buffer = {}
        
        def extract_RBFHead_params(layer, idx, classes):
            if type(layer).__name__ == "SingleRBFHeadLayer":
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
                    if type(layer).__name__ == "SingleRBFHeadLayer":
                        self.params_buffer.update(extract_RBFHead_params(layer, "head", self.old_nclasses))
                    else:
                        # TODO Extend to other classifiers
                        raise ValueError("Deprecation: The classifier head must be SingleRBFHeadLayer.")

            with torch.no_grad():   
                self.old_module = deepcopy(self.module)
                for p in self.old_module.parameters():
                    p.requires_grad = False
                self.old_module.eval()

            self.generate_dreamed_data()
            # self.visualize_dreamed_data()

    def generate_dreamed_data(self):
        """
        Generate diverse and realistic dreamed images by ensuring:
        - Activation alignment with Gaussian centers.
        - Sufficient intra-class diversity.
        - Proper class separation.
        """

        self.old_module.eval()
        kernels_centers = self.params_buffer["kernels_centers_head"]
        sigmas = self.params_buffer["log_shapes_head"].exp()  

        self.dreamed_data_buffer = {}

        in_shape = self.in_shape
        batch_size = self.no_samples_per_class
        steps = self.steps
        lambda_l2 = self.lambda_l2
        lambda_tv = self.lambda_tv
        dreaming_lr = self.dreaming_lr
        diversity_weight = self.diversity_weight

        for class_idx, (mean, cov) in enumerate(zip(kernels_centers, sigmas)):
            dreamed_images = torch.randn((batch_size, *in_shape), requires_grad=True)

            optimizer = torch.optim.Adam([dreamed_images], lr=dreaming_lr)

            for _ in range(steps):
                optimizer.zero_grad()

                _ = self.old_module(dreamed_images)
                stored_activations = self.old_module.activations[-1]

                act_mean = stored_activations.mean(dim=0)
                act_var = stored_activations.var(dim=0, unbiased=False)

                # Loss: Match activation mean & variance
                mean_loss = F.mse_loss(act_mean, mean)
                cov_loss = F.mse_loss(act_var, cov)

                # Mahalanobis Distance (Relaxed)
                diff = stored_activations - mean
                mahalanobis_loss = torch.mean((diff / (cov + 1e-6)) ** 2)  # Add small value to prevent collapse

                # L2 Prior (Keeps values reasonable)
                l2_loss = torch.norm(dreamed_images, p=2) / batch_size

                # Total Variation Loss (Smoothness)
                tv_loss = torch.mean(torch.abs(dreamed_images[:, :, :-1, :] - dreamed_images[:, :, 1:, :])) + \
                        torch.mean(torch.abs(dreamed_images[:, :, :, :-1] - dreamed_images[:, :, :, 1:]))

                # Contrastive Diversity Loss
                diversity_loss = -torch.mean(torch.norm(stored_activations.unsqueeze(1) - stored_activations.unsqueeze(0), dim=-1))

                # Final Loss
                total_loss = mean_loss + cov_loss + mahalanobis_loss + lambda_l2 * l2_loss + lambda_tv * tv_loss + \
                    diversity_weight * diversity_loss

                # Backpropagate
                total_loss.backward()
                optimizer.step()

            # Store dreamed samples
            dreamed_images = dreamed_images.detach()
            self.dreamed_data_buffer[class_idx] = dreamed_images

    def visualize_dreamed_data(self, num_classes=2, num_samples=5):
        """
        Visualizes a few samples of dreamed data from the first two classes.
        
        Args:
            num_classes (int): Number of classes to visualize (default: 2).
            num_samples (int): Number of samples per class to visualize (default: 5).
        """
        # Select only the first `num_classes` entries
        selected_classes = list(self.dreamed_data_buffer.keys())[:num_classes]
        
        fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 2, num_classes * 2))

        for row, class_idx in enumerate(selected_classes):
            dreamed_images = self.dreamed_data_buffer[class_idx]
            for col in range(num_samples):
                img = dreamed_images[col].squeeze(0).numpy()  # Remove channel dim for grayscale
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')
                if col == 0:
                    axes[row, col].set_title(f"Class {class_idx}")

        plt.suptitle("Dreamed Data Visualization")
        plt.show()
    

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
            feature_distillation_loss = 0.0
            num_layers = len(self.old_module.activations)

            for class_idx in self.dreamed_data_buffer.keys():
                dreamed_images = self.dreamed_data_buffer[class_idx]

                _ = self.old_module(dreamed_images)
                old_activations = [act.detach() for act in self.old_module.activations]

                _ = self.module(dreamed_images)
                curr_activations = self.module.activations

                for old_act, curr_act in zip(old_activations, curr_activations):
                    feature_distillation_loss += torch.norm(old_act - curr_act, p=2)

            feature_distillation_loss = feature_distillation_loss / (num_layers * len(self.dreamed_data_buffer))

            loss = loss + self.alpha * feature_distillation_loss

        return loss, preds
