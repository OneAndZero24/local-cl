"""
Regression head for continual learning models.
"""

import torch
from torch import nn
import torch.nn.functional as F

from model.layer import instantiate, LayerType
from model.layer.interval_activation import IntervalActivation


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
        use_interval_activation: bool = True,
        activation: str = 'relu',
        **kwargs,
    ):
        """
        Initializes the RegressionHead.
        
        Args:
            in_features (int): Number of input features.
            out_features (int, optional): Number of output features. Defaults to 1.
            layer_type (LayerType, optional): Type of layer to use. Defaults to LayerType.NORMAL.
            use_interval_activation (bool, optional): Whether to add IntervalActivation after regressor. Defaults to True.
            activation (str, optional): Activation function to apply before IntervalActivation. Options: 'none', 'relu', 'tanh', 'sigmoid', 'sin'. Defaults to 'none'.
            **kwargs: Additional keyword arguments to pass to the layer instantiation.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_interval_activation = use_interval_activation
        self.activation_type = activation
        
        self.regressor = instantiate(
            layer_type,
            in_features,
            out_features,
            **kwargs
        )
        
        # Add IntervalActivation after the regressor to track output intervals
        # Don't apply activation - just track the raw regression output values
        if use_interval_activation:
            self.interval_activation = IntervalActivation(
                lower_percentile=0.05,
                upper_percentile=0.95,
                log_name='regression_output_interval',
                apply_activation=False  # Don't modify the output, just track intervals
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the regressor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """
        out = self.regressor(x)
        
        # DEBUG: Track batch count
        if not hasattr(self, '_debug_batch_count'):
            self._debug_batch_count = 0
        self._debug_batch_count += 1
        
        if self._debug_batch_count % 50 == 1 and self.training:
            print(f"\n[REGRESSION_HEAD DEBUG] Batch {self._debug_batch_count}")
            print(f"  Input to head range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"  Raw regressor output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        # Apply activation function if specified
        if self.activation_type == 'relu':
            out = F.relu(out)
        elif self.activation_type == 'tanh':
            out = torch.tanh(out)
        elif self.activation_type == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation_type == 'sin':
            out = torch.sin(out)
        elif self.activation_type == 'cos':
            out = torch.cos(out)
        # 'none' - no activation applied
        
        if self._debug_batch_count % 50 == 1 and self.training and self.activation_type != 'none':
            print(f"  After '{self.activation_type}' activation: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        if self.use_interval_activation:
            out = self.interval_activation(out)
            
            if self._debug_batch_count % 50 == 1 and self.training:
                print(f"  After IntervalActivation (apply_activation={self.interval_activation.apply_activation}): [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        return out
