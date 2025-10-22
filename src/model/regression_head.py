"""
Regression head for continual learning models.
"""

import torch
from torch import nn

from model.layer import instantiate, LayerType


class RegressionHead(nn.Module):
    """
    Simple regression head that outputs continuous values.
    
    Unlike IncrementalClassifier, this head has a fixed output dimension
    and does not need incremental updates.
    
    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features (usually 1 for regression).
        regressor (nn.Module): The regression layer.
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the regressor.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        layer_type: LayerType = LayerType.NORMAL,
        **kwargs,
    ):
        """
        Initializes the RegressionHead.
        
        Args:
            in_features (int): Number of input features.
            out_features (int, optional): Number of output features. Defaults to 1.
            layer_type (LayerType, optional): Type of layer to use. Defaults to LayerType.NORMAL.
            **kwargs: Additional keyword arguments to pass to the layer instantiation.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.regressor = instantiate(
            layer_type,
            in_features,
            out_features,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the regressor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """
        return self.regressor(x)
