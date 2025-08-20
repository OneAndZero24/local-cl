import torch
import torch.nn as nn
from torchvision import models
from .cl_module_abc import CLModuleABC
from .layer.interval_activation import IntervalActivation
from .inc_classifier import IncrementalClassifier


class ResNet18Interval(CLModuleABC):
    """
    ResNet18 with IntervalActivation layers inserted between each layer.
    """

    def __init__(
        self,
        initial_out_features: int,
        pretrained: bool = True,
        frozen: bool = False,
        interval_layer_kwargs: dict = None,
        head_type: str = "Normal",
        head_kwargs: dict = {}
    ):
        """
        Initialize ResNet18 with interval activation layers.

        Args:
            initial_out_features (int): Initial number of output classes
            pretrained (bool): Whether to use pretrained weights
            frozen (bool): Whether to freeze the backbone
            interval_layer_kwargs (dict): Arguments for IntervalActivation layers
            head_type (str): Type of incremental classifier head
            head_kwargs (dict): Additional arguments for the head
        """
        super().__init__(
            IncrementalClassifier(
                in_features=512,  # ResNet18's final feature dimension
                initial_out_features=initial_out_features,
                head_type=head_type,
                **head_kwargs
            )
        )
        
        if interval_layer_kwargs is None:
            interval_layer_kwargs = {
                'lower_percentile': 0.05,
                'upper_percentile': 0.95
            }

        base_model = models.resnet18(pretrained=pretrained)
        
        layers = []
        
        # Calculate feature map sizes based on input
        H = W = 32  # MNIST/CIFAR input size
        
        # Initial conv: stride=2, kernel=7, padding=3
        H = (H + 2*3 - 7)//2 + 1  # 16
        W = (W + 2*3 - 7)//2 + 1  # 16
        
        # Maxpool: stride=2, kernel=3, padding=1
        H = (H + 2*1 - 3)//2 + 1  # 8
        W = (W + 2*1 - 3)//2 + 1  # 8
        
        # Initial convolution block
        layers.extend([
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            IntervalActivation((64, H, W), log_name='interval_stem', **interval_layer_kwargs)
        ])
        
        # Layer1 (64 channels) - maintains resolution
        layers.extend([
            base_model.layer1,
            IntervalActivation((64, H, W), log_name='interval_layer1', **interval_layer_kwargs)
        ])
        
        # Layer2 (128 channels) - halves resolution
        H, W = H//2, W//2  # 4x4
        layers.extend([
            base_model.layer2,
            IntervalActivation((128, H, W), log_name='interval_layer2', **interval_layer_kwargs)
        ])
        
        # Layer3 (256 channels) - halves resolution
        H, W = H//2, W//2  # 2x2
        layers.extend([
            base_model.layer3,
            IntervalActivation((256, H, W), log_name='interval_layer3', **interval_layer_kwargs)
        ])
        
        # Layer4 (512 channels) - halves resolution
        H, W = H//2, W//2  # 1x1
        layers.extend([
            base_model.layer4,
            IntervalActivation((512, H, W), log_name='interval_layer4', **interval_layer_kwargs)
        ])
        
        # Final layers
        layers.extend([
            base_model.avgpool,
            nn.Flatten()
        ])
        
        self.layers = nn.ModuleList(layers)
        
        if frozen:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all parameters except the final classifier layer."""
        for name, param in self.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False

    def forward(self, x):
        """Forward pass with interval activations between layers."""
        self.reset_activations()
        
        # Handle grayscale images by repeating the channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, IntervalActivation):
                self.add_activation(layer, x)
                
        return self.head(x)
