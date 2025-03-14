import logging
import torch
from src.method.method_plugin_abc import MethodPluginABC
from copy import deepcopy
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class PenultimateLayerReg(MethodPluginABC):
    """
    This class applies regularization to the penultimate layer to mitigate feature drift.
    It is relevant only when SingleRBFHead/MultiRBFHead is used.
    
    Attributes:
        params_buffer (dict): Stores model parameters for regularization.
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float):
        """
        Initializes the feature drift regularization method for the penultimate layer.

        Args:
            alpha (float): Regularization coefficient.
        """
        warnings.warn(
            "PenultimateLayerReg is deprecated and may be removed in future versions. "
            "Consider using updated methods for regularization.",
            DeprecationWarning,
            2
        )
        
        super().__init__()
        self.params_buffer = {}

        self.alpha = alpha
        log.info(f"PenultimateLayerReg initialized with alpha={alpha}")

    def setup_task(self, task_id: int):
        """
        Sets up a new task and stores parameters from the penultimate layer
        and incremental classification head (RBF).

        Args:
            task_id (int): Unique task identifier.
        """
        # Warning about deprecation
        warnings.warn(
            "PenultimateLayerReg is deprecated and may be removed in future versions. "
            "Consider using updated methods for regularization.",
            DeprecationWarning,
            2
        )

        def extract_layer_params(layer, idx, classes=None):
            if type(layer).__name__ in ["RBFLayer", "RBFHeadLayer"]:
                weights = layer.weights.detach().clone() if classes is None else None
                kernels_centers = layer.kernels_centers[:classes, :].detach().clone() if classes is not None else layer.kernels_centers.detach().clone()
                log_shapes = layer.log_shapes[:classes, :].detach().clone() if classes is not None else layer.log_shapes.detach().clone()
                return {
                    f"weights_{idx}": weights,
                    f"kernels_centers_{idx}": kernels_centers,
                    f"log_shapes_{idx}": log_shapes
                }
            elif isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().clone()
                biases = layer.bias.detach().clone() if layer.bias is not None else None
                return {
                    f"linear_weights_{idx}": weights,
                    f"linear_biases_{idx}": biases
                }
            return {}

        self.task_id = task_id

        if task_id > 0:
            model = self.module.module
            for module in model.children():
                if isinstance(module, torch.nn.ModuleList):
                    for idx, layer in enumerate(module):
                        self.params_buffer.update(extract_layer_params(layer, idx))
                elif type(module).__name__ == "IncrementalClassifier":
                    layer = module.classifier
                    old_nclasses = module.old_nclasses
                    if type(layer).__name__ == "RBFHeadLayer":
                        self.params_buffer.update(extract_layer_params(layer, "head", old_nclasses))

            with torch.no_grad():   
                self.old_module = deepcopy(self.module)
                for p in self.old_module.parameters():
                    p.requires_grad = False
                self.old_module.eval()

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
