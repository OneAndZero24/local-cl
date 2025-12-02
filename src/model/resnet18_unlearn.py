import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .layer.interval_activation import IntervalActivation


class ResNet18Unlearn(nn.Module):
    """
    Simple ResNet18 with interval activation before the final classifier.
    Uses pretrained torchvision ResNet18 with 10-class output for CIFAR-10.
    
    Architecture:
        ResNet18 backbone (pretrained) -> IntervalActivation -> Linear(512, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ):
        """
        Initialize ResNet18 for unlearning experiments.
        
        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            pretrained: Whether to use ImageNet pretrained weights (default: True)
            lower_percentile: Lower percentile for interval activation (default: 0.05)
            upper_percentile: Upper percentile for interval activation (default: 0.95)
        """
        super().__init__()
        
        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Remove the final FC layer
        self.backbone.fc = nn.Identity()
        
        # Add interval activation layer before classifier
        self.interval_activation = IntervalActivation(
            layer_name="pre_classifier",
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            use_nonlinear_transform=False  # ResNet already has ReLU
        )
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        # Extract features
        x = self.backbone(x)
        
        # Apply interval activation
        x = self.interval_activation(x)
        
        # Classify
        x = self.classifier(x)
        
        return x
