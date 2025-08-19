import torch
import torch.nn as nn
from torchvision import models
from .cl_module_abc import CLModuleABC
from .layer.interval_activation import IntervalActivation


class ResNet18Interval(CLModuleABC):
    """
    ResNet18 with IntervalActivation layers inserted between each layer.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        frozen: bool = False,
        interval_layer_kwargs: dict = None
    ):
        """
        Initialize ResNet18 with interval activation layers.

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            frozen (bool): Whether to freeze the backbone
            interval_layer_kwargs (dict): Arguments for IntervalActivation layers
        """
        super().__init__(None)  # We'll set the head later
        
        # Default kwargs for interval layers if none provided
        if interval_layer_kwargs is None:
            interval_layer_kwargs = {
                'lower_percentile': 0.05,
                'upper_percentile': 0.95
            }

        # Get the pretrained model
        base_model = models.resnet18(pretrained=pretrained)
        
        # Create the feature extractor parts with interval activations
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.interval1 = IntervalActivation((64, 56, 56), name='interval_stem', **interval_layer_kwargs)
        
        # Layer1 (64 channels)
        self.layer1 = base_model.layer1
        self.interval2 = IntervalActivation((64, 56, 56), name='interval_layer1', **interval_layer_kwargs)
        
        # Layer2 (128 channels)
        self.layer2 = base_model.layer2
        self.interval3 = IntervalActivation((128, 28, 28), name='interval_layer2', **interval_layer_kwargs)
        
        # Layer3 (256 channels)
        self.layer3 = base_model.layer3
        self.interval4 = IntervalActivation((256, 14, 14), name='interval_layer3', **interval_layer_kwargs)
        
        # Layer4 (512 channels)
        self.layer4 = base_model.layer4
        self.interval5 = IntervalActivation((512, 7, 7), name='interval_layer4', **interval_layer_kwargs)
        
        self.avgpool = base_model.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
        if frozen:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all parameters except the final classifier layer."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass with interval activations between layers."""
        self.reset_activations()
        
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.interval1(x)
        self.add_activation(self.interval1, x)
        
        # ResNet blocks with interval activations
        x = self.layer1(x)
        x = self.interval2(x)
        self.add_activation(self.interval2, x)
        
        x = self.layer2(x)
        x = self.interval3(x)
        self.add_activation(self.interval3, x)
        
        x = self.layer3(x)
        x = self.interval4(x)
        self.add_activation(self.interval4, x)
        
        x = self.layer4(x)
        x = self.interval5(x)
        self.add_activation(self.interval5, x)
        
        # Final layers
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

    def reset_intervals(self):
        """Reset all interval activation layers."""
        self.interval1.reset_range()
        self.interval2.reset_range()
        self.interval3.reset_range()
        self.interval4.reset_range()
        self.interval5.reset_range()
