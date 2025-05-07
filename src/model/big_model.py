import torch
import torch.nn as nn
from torchvision import models

from model.cl_module_abc import CLModuleABC


class BigModel(CLModuleABC):
    """
    A custom model that combines a pretrained backbone (ResNet-18, ResNet-50, or ViT-B/16) from torchvision with a custom head module.
    
    Args:
        head (nn.Module): The custom head module to apply after feature extraction.
        pretrained_backbone_name (str): The name of the pretrained backbone model ('resnet18', 'resnet50', or 'vit_b_16').
        pretrained (bool, optional): Whether to use a pretrained backbone. Defaults to True.
        frozen (bool, optional): Whether to freeze the backbone parameters. Defaults to False.
        size (tuple[int], optional): The target size for input tensors (height, width). Defaults to (224, 224).
        reduced_dim (int, optional): Output feature dimension after replacing the last ResNet block. Defaults to 128.
    
    Attributes:
        fe (nn.Module): The feature extractor backbone model (ResNet-18, ResNet-50, or ViT-B/16).
        reducer (nn.Module): Lightweight custom layer replacing the last ResNet block.
        head (nn.Module): The custom head module.
        layers (list[nn.Module]): The layers of the custom head module.
        size (tuple[int]): The target size for input tensors.
    
    Methods:
        transform(tensor):
            Transforms the input tensor to ensure it has the correct shape and size for the model.
        forward(x):
            Performs a forward pass through the model.
    """

    def __init__(self,
        head: nn.Module,
        pretrained_backbone_name: str,
        pretrained: bool=True,
        frozen: bool=False,
        size: tuple[int]=(224, 224),
        reduced_dim: int=64,
    ):
        """
        Initializes the BigModel with a pretrained backbone and a custom head.
        
        Args:
            head (nn.Module): The custom head module to apply after feature extraction.
            pretrained_backbone_name (str): The name of the pretrained backbone model ('resnet18', 'resnet50', or 'vit_b_16').
            pretrained (bool, optional): Whether to use a pretrained backbone. Defaults to True.
            frozen (bool, optional): Whether to freeze the backbone parameters. Defaults to False.
            size (tuple[int], optional): The target size for input tensors (height, width). Defaults to (224, 224).
            reduced_dim (int, optional): Output feature dimension after replacing the last ResNet block. Defaults to 64.
        """
        super().__init__(head.head)

        self.flatten_output = False
        self.reducer = None
        self.output_dim = reduced_dim

        if pretrained_backbone_name not in ['resnet18', 'resnet50', 'vit_b_16']:
            raise ValueError("pretrained_backbone_name must be 'resnet18', 'resnet50', or 'vit_b_16'")

        channels = None
        if pretrained_backbone_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            channels = 256
            self.fe = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
            )
            self.reducer = nn.Sequential(
                nn.Conv2d(channels, reduced_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        elif pretrained_backbone_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            channels = 1024
            self.fe = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
            )
            self.reducer = nn.Sequential(
                nn.Conv2d(channels, reduced_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        else:
            vit = models.vit_b_16(pretrained=pretrained)
            self.fe = nn.Sequential(*list(vit.children())[:-1])
            self.flatten_output = False
            self.reducer = nn.Identity()

        if (channels is not None) and (channels == reduced_dim):
            self.reducer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.c_head = head
        self.layers = head.layers
        self.size = size
        self.frozen = frozen

        if self.frozen:
            self.fe.eval()
            for param in self.fe.parameters():
                param.requires_grad = False

            for param in self.reducer.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Performs a forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor to process. Expected shape is [C, H, W] or [H, W].
        
        Returns:
            torch.Tensor: The output of the custom head after processing the input tensor.
        """  
        self.reset_activations()

        if self.frozen:
            with torch.no_grad():
                x = self.fe(x)
        else:
            x = self.fe(x)

        x = self.reducer(x)
        return self.c_head(x)
