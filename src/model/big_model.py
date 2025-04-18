import timm
import torch.nn as nn
from torch import functional as F

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
        out_index: int=-1,
        size: tuple[int]=(224, 224),
    ):
        """
        Initializes the BigModel with a pretrained backbone and a custom head.
        
        Args:
            pretrained_backbone_name (str): The name of the pretrained backbone model to use.
            head (nn.Module): The custom head module to apply after feature extraction.
            out_index (int, optional): The index of the output feature map from the backbone. Defaults to -1.
            size (tuple[int], optional): The target size for input tensors (height, width). Defaults to (224, 224).
        """

        self.fe = timm.create_model(
            pretrained_backbone_name, 
            features_only=True,
            out_indices=(out_index),
            pretrained=True
        )
        self.head = head
        self.layers = head.layers
        self.size = size

        super().__init__(self.head.head)

    def transform(self, tensor):
        """
        Transforms the input tensor to ensure it has the correct shape and size for the model.
        
        Args:
            tensor (torch.Tensor): The input tensor to transform. Expected shape is [C, H, W] or [H, W].
        
        Returns:
            torch.Tensor: The transformed tensor with shape [3, H, W] and resized to the target size.
        
        Raises:
            ValueError: If the input tensor has an unexpected number of channels.
        """

        # Ensure it's [C, H, W]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # [H, W] -> [1, H, W]

        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        elif tensor.size(0) != 3:
            raise ValueError(f"Unexpected number of channels: {tensor.size(0)}")

        # Add batch dim for interpolate: [1, 3, H, W]
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)  # Back to [3, H, W]

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
        x = self.fe(x)
        return self.head(x)