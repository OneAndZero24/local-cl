import torch
import torch.nn as nn
from torch.nn import functional as F
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
    
    Attributes:
        fe (nn.Module): The feature extractor backbone model (ResNet-18, ResNet-50, or ViT-B/16).
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
    ):
        """
        Initializes the BigModel with a pretrained backbone and a custom head.
        
        Args:
            head (nn.Module): The custom head module to apply after feature extraction.
            pretrained_backbone_name (str): The name of the pretrained backbone model ('resnet18', 'resnet50', or 'vit_b_16').
            pretrained (bool, optional): Whether to use a pretrained backbone. Defaults to True.
            frozen (bool, optional): Whether to freeze the backbone parameters. Defaults to False.
            size (tuple[int], optional): The target size for input tensors (height, width). Defaults to (224, 224).
        """

        super().__init__(head.head)

        if pretrained_backbone_name not in ['resnet18', 'resnet50', 'vit_b_16']:
            raise ValueError("pretrained_backbone_name must be 'resnet18', 'resnet50', or 'vit_b_16'")

        if pretrained_backbone_name == 'resnet18':
            self.fe = models.resnet18(pretrained=pretrained)
            self.fe = nn.Sequential(*list(self.fe.children())[:-1])  # Remove FC layer
            self.flatten_output = True
        elif pretrained_backbone_name == 'resnet50':
            self.fe = models.resnet50(pretrained=pretrained)
            self.fe = nn.Sequential(*list(self.fe.children())[:-1])  # Remove FC layer
            self.flatten_output = True
        else: 
            self.fe = models.vit_b_16(pretrained=pretrained)
            self.fe = nn.Sequential(*list(self.fe.children())[:-1])  # Remove classification head
            self.flatten_output = False

        self.c_head = head
        self.layers = head.layers
        self.size = size
        self.frozen = frozen

        if self.frozen:
            self.fe.eval()
            for param in self.fe.parameters():
                param.requires_grad = False

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
        
        if self.flatten_output:
            x = x.view(x.size(0), -1)
        
        return self.c_head(x)