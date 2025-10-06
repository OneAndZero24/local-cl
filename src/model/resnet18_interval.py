import torch.nn as nn
import torch
from torchvision import models

from .cl_module_abc import CLModuleABC
from .layer.interval_activation import IntervalActivation
from .inc_classifier import IncrementalClassifier



class ResNet18Interval(CLModuleABC):
    """
    ResNet18 backbone augmented with IntervalActivation layers and an MLP head.

    This model uses a standard ResNet18 feature extractor and inserts two
    `IntervalActivation` layers:
        1. After the feature extraction and flattening (end of reducer).
        2. After a hidden linear layer in the MLP before the classifier.

    The final classification is performed via an `IncrementalClassifier` head.

    Attributes:
        fe_layers (list[nn.Module]): List of ResNet18 feature extraction layers.
        mlp_layers (list[nn.Module]): List of MLP layers including hidden Linear and IntervalActivation layers.
        layers (nn.ModuleList): Combined feature extractor and MLP layers.
        head (IncrementalClassifier): Incremental classifier head.
    """

    def __init__(
        self,
        initial_out_features: int,
        dim_hidden: int,
        frozen: bool = False,
        interval_layer_kwargs: dict = None,
        head_type: str = "Normal",
        mask_past_classifier_neurons: bool = True,
        head_kwargs: dict = {}
    ) -> None:
        """
        Initialize ResNet18Interval model.

        Args:
            initial_out_features (int): Number of output classes for the classifier.
            dim_hidden (int): Number of hidden units in the MLP layer following the ResNet18 backbone.
            frozen (bool): If True, freeze the backbone (ResNet18) parameters. Default is False.
            interval_layer_kwargs (dict): Arguments for IntervalActivation layers 
                                          (e.g., {'lower_percentile': 0.05, 'upper_percentile': 0.95}).
            head_type (str): Type of incremental classifier head. Default is "Normal".
            mask_past_classifier_neurons (boolk): Flag to indicate whether mask classifier neurons for old classes.
                                                    Default is True.
            head_kwargs (dict): Additional keyword arguments for the incremental classifier.
        """
        
        self.frozen = frozen

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

        self.fe = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final fully-connected (classification) layer from the ResNet model
        self.fe.fc = nn.Identity() 
    
        mlp_layers = [
            IntervalActivation(512, **interval_layer_kwargs),
            nn.Linear(512, dim_hidden),
            IntervalActivation(dim_hidden, **interval_layer_kwargs) 
        ]
        self.layers = nn.ModuleList(mlp_layers)
                
        if self.frozen:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters in the ResNet18 backbone to prevent them from updating
        during training.

        Only the parameters in the MLP and the IncrementalClassifier head remain trainable.
        """
        for param in self.fe.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        The input `x` passes sequentially through:
            1. The ResNet18 feature extractor (`fe_layers`).
            2. The MLP layers with IntervalActivation (`mlp_layers`).
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
        for layer in self.layers:
            x = layer(x)
                
        return self.head(x)
