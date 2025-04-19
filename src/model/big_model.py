import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.cl_module_abc import CLModuleABC


class BigModel(CLModuleABC):
    """
    A custom model that combines a pretrained backbone from the `timm` library with a custom head module.
    
    Args:
        pretrained_backbone_name (str): The name of the pretrained backbone model to use (eg. resnet50, vit_base_patch16_224_in21k).
        head (nn.Module): The custom head module to apply after feature extraction.
        out_index (int, optional): The index of the output feature map from the backbone. Defaults to -1.
        size (tuple[int], optional): The target size for input tensors (height, width). Defaults to (224, 224).
    
    Attributes:
        fe (nn.Module): The feature extractor backbone model.
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
        pretrained_backbone_name: str,
        head: nn.Module,
        frozen: bool=True,
        out_index: int=-1,
        size: tuple[int]=(224, 224),
    ):
        """
        Initializes the BigModel with a pretrained backbone and a custom head.
        
        Args:
            pretrained_backbone_name (str): The name of the pretrained backbone model to use.
            head (nn.Module): The custom head module to apply after feature extraction.
            out_index (int, optional): The index of the output feature map from the backbone. Defaults to -1.
            frozen (bool, optional): Whether to freeze the backbone parameters. Defaults to True.
            size (tuple[int], optional): The target size for input tensors (height, width). Defaults to (224, 224).
        """

        super().__init__(head.head)

        self.fe = timm.create_model(
            pretrained_backbone_name, 
            features_only=True,
            pretrained=True
        )

        self.head = head
        self.layers = head.layers
        self.size = size
        self.out_index = out_index
        self.frozen = frozen

        if self.frozen:
            self.fe.eval()

    def transform(self, tensor):
        """
        Transforms the input tensor to ensure it has the correct shape and size for the model.

        Args:
            tensor (torch.Tensor): The input tensor to transform. 
                Expected shapes: [H, W], [C, H, W], or [B, C, H, W].

        Returns:
            torch.Tensor: The transformed tensor with shape [B, 3, H, W], resized to target size.

        Raises:
            ValueError: If the input tensor has an unexpected number of channels.
        """

        if tensor.ndim == 2:       # If [H, W], add batch and channel dimensions
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        elif tensor.ndim == 3:     # If [C, H, W], add batch dimension
            tensor = tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        elif tensor.ndim != 4:     # Now tensor is [B, C, H, W] 
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        b, c, h, w = tensor.shape

        if c == 1:
            tensor = tensor.repeat(1, 3, 1, 1)  # Grayscale to RGB
        elif c != 3:
            raise ValueError(f"Unexpected number of channels: {c}")

        # Resize to target size
        tensor = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)

        return tensor

    def forward(self, x):
        """
        Performs a forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor to process. Expected shape is [C, H, W] or [H, W].
        
        Returns:
            torch.Tensor: The output of the custom head after processing the input tensor.
        """  
        x = self.transform(x)
        self.reset_activations()
        if self.frozen:
            with torch.no_grad():
                x = self.fe(x)[self.out_index]
        else:   
            x = self.fe(x)[self.out_index]
        return self.head(x)