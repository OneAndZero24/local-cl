import torch.nn as nn
import torch
from torchvision import models

from .cl_module_abc import CLModuleABC
from .inc_classifier import IncrementalClassifier


class ResNet18(CLModuleABC):
    """
    ResNet18 backbone with an MLP head (no IntervalActivation layers).

    Equivalent to ResNet18Interval but without IntervalActivation layers.

    Attributes:
        fe (nn.Module): ResNet18 feature extractor (up to the last fully connected layer).
        layers (nn.ModuleList): MLP with one hidden Linear layer before the classifier.
        head (IncrementalClassifier): Incremental classifier head.
    """

    def __init__(
        self,
        initial_out_features: int,
        dim_hidden: int,
        head_type: str = "Normal",
        mask_past_classifier_neurons: bool = False,
        head_kwargs: dict = {}
    ) -> None:
        """
        Initialize ResNet18 model.

        Args:
            initial_out_features (int): Number of output classes for the classifier.
            dim_hidden (int): Number of hidden units in the MLP layer following the ResNet18 backbone.
            head_type (str): Type of incremental classifier head. Default is "Normal".
            mask_past_classifier_neurons (bool): Whether to mask classifier neurons for old classes.
            head_kwargs (dict): Additional keyword arguments for the incremental classifier.
        """
        head = IncrementalClassifier(
            in_features=dim_hidden,
            initial_out_features=initial_out_features,
            head_type=head_type,
            mask_past_classifier_neurons=mask_past_classifier_neurons,
            **head_kwargs
        )
        super().__init__(head)

        self.fe = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.fe.fc = nn.Identity()

        mlp_layers = [
            nn.Linear(512, dim_hidden),
            nn.ReLU(inplace=True)
        ]
        self.layers = nn.ModuleList(mlp_layers)

        # Freeze all layers except the last ResNet block
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters in the ResNet18 backbone except the final block (`layer4`).
        """
        for name, param in self.fe.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor, shape [B, 3, H, W].

        Returns:
            torch.Tensor: Output logits.
        """
        self.reset_activations()

        x = self.fe(x)

        for layer in self.layers:
            x = layer(x)

        x = self.head(x)
        return x
