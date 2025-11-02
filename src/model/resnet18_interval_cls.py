import torch.nn as nn
import torch
from torchvision import models

from .cl_module_abc import CLModuleABC
from .layer.interval_activation import IntervalActivation
from .inc_classifier import IncrementalClassifier

class ResNet18IntervalCls(CLModuleABC):
    """
    ResNet18 backbone augmented with IntervalActivation layers in an MLP head.

    This model uses a standard ResNet18 feature extractor and inserts two
    `IntervalActivation` layers:
        1. After the feature extraction and flattening (end of reducer).
        2. After a hidden linear layer in the MLP before the classifier.

    The final classification is performed via an `IncrementalClassifier` head.

    Attributes:
        fe (nn.Module): ResNet18 feature extractor (up to the last fully connected layer).
        mlp (nn.ModuleList): MLP with two IntervalActivation layers.
        head (IncrementalClassifier): Incremental classifier head.
    """

    def __init__(
        self,
        initial_out_features: int,
        dim_hidden: int,
        interval_layer_kwargs: dict = None,
        head_type: str = "Normal",
        mask_past_classifier_neurons: bool = False,
        head_kwargs: dict = {}
    ) -> None:
        """
        Initialize ResNet18Interval model.

        Args:
            initial_out_features (int): Number of output classes for the classifier.
            dim_hidden (int): Number of hidden units in the MLP layer following the ResNet18 backbone.
            interval_layer_kwargs (dict): Arguments for IntervalActivation layers 
                                          (e.g., {'lower_percentile': 0.05, 'upper_percentile': 0.95}).
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
        
        if interval_layer_kwargs is None:
            interval_layer_kwargs = {
                'lower_percentile': 0.05,
                'upper_percentile': 0.95
            }

        # Load pretrained ResNet18
        self.fe = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.fe.fc = nn.Identity() 

        mlp_layers = [
            IntervalActivation(**interval_layer_kwargs),
            nn.Linear(512, dim_hidden),
            IntervalActivation(**interval_layer_kwargs)
        ]
        self.mlp = nn.ModuleList(mlp_layers)

        # Freeze everything except the last ResNet block
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters in the ResNet18 backbone except the final block (`layer4`).
        This keeps low-level features stable while allowing high-level adaptation.
        """
        for name, param in self.fe.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        The input `x` passes sequentially through:
            1. The ResNet18 feature extractor.
            2. The MLP layers with IntervalActivation (`mlp`).
            3. The IncrementalClassifier head (`self.head`).

        Grayscale images (1-channel) are automatically repeated to 3 channels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits from the classifier head.
        """
        
        # Handle grayscale images by repeating the channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = self.fe(x)
        for layer in self.mlp:
            x = layer(x)
                
        return self.head(x)