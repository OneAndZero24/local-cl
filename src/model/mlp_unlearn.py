import torch
import torch.nn as nn

from .layer.interval_activation import IntervalActivation


class MLPUnlearn(nn.Module):
    """
    Simple MLP with interval activation before the final classifier for MNIST unlearning.
    
    Architecture:
        Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> 
        IntervalActivation -> Linear(128, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        hidden_sizes: list = [256, 128],
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ):
        """
        Initialize MLP for unlearning experiments.
        
        Args:
            num_classes: Number of output classes (default: 10 for MNIST)
            hidden_sizes: List of hidden layer sizes (default: [256, 128])
            lower_percentile: Lower percentile for interval activation (default: 0.05)
            upper_percentile: Upper percentile for interval activation (default: 0.95)
        """
        super().__init__()
        
        # Input size for MNIST (28x28)
        input_size = 784
        
        # Build hidden layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Add interval activation layer before classifier
        self.interval_activation = IntervalActivation(
            layer_name="pre_classifier",
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            use_nonlinear_transform=False  # Already have ReLU in hidden layers
        )
        
        # Final classifier
        self.classifier = nn.Linear(prev_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 1, 28, 28] or [B, 784]
            
        Returns:
            Logits [B, num_classes]
        """
        # Flatten if needed
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Apply interval activation
        x = self.interval_activation(x)
        
        # Classify
        x = self.classifier(x)
        
        return x
