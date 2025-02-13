import torch
import torch.nn as nn

from model.cl_module_abc import CLModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer import instantiate, LayerType


class MLP(CLModuleABC):
    """
    Multi-Layer Perceptron (MLP) class that extends CLModuleABC.

    Args:
        initial_out_features (int): The number of output features for the initial layer.
        sizes (list[int]): A list of integers representing the sizes of each layer.
        layers (list[str]): A list of strings representing the type of each layer.
        head_type (str, optional): The type of the head layer. Defaults to "Normal".
        activation (nn.Module, optional): The activation function to use between layers. Defaults to nn.Tanh().
        **kwargs: Additional keyword arguments for layer instantiation.

    Attributes:
        layers (nn.ModuleList): A list of instantiated layers.

    Methods:
        forward(x):
            Performs a forward pass through the network and records activations.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the head layer.
    """

    def __init__(self,
        initial_out_features: int,
        sizes: list[int],
        layers: list[str],
        head_type: str="Normal",
        activation: nn.Module=nn.Tanh(),
        **kwargs
    ):
        """
        Initializes the MLP model.
        
        Args:
            initial_out_features (int): The initial number of output features.
            sizes (list[int]): A list of integers representing the sizes of each layer.
            layers (list[str]): A list of strings representing the types of each layer.
            head_type (str, optional): The type of the head layer. Defaults to "Normal".
            activation (nn.Module, optional): The activation function to use between layers. Defaults to nn.Tanh().
            **kwargs: Additional keyword arguments for layer instantiation.

        Keyword Args:
            train_head_domain (bool, optional): If head_type is LOCAL, specifies whether to train the head domain. Defaults to False.
            masking (optional): Masking parameter for layers.
            mask_value (optional): Mask value parameter for layers.
        """
                
        assert len(sizes)-1 == len(layers), "Number of sizes and layers must match"

        head_type = LayerType(head_type)
        layer_types = list(map(lambda x: LayerType(x), layers))

        head_kwargs = kwargs.copy()
        if head_type == LayerType.LOCAL:
            head_kwargs["train_domain"] = kwargs.get("train_head_domain", False)
            head_kwargs.pop("train_head_domain", None)
        kwargs.pop("train_head_domain", None)

        super().__init__(
            IncrementalClassifier(
                sizes[-1], 
                initial_out_features,
                head_type,
                **head_kwargs
            )
        )

        kwargs.pop("masking", None)
        kwargs.pop("mask_value", None)

        layers = []
        N = len(sizes)-1
        for i in range(N):                    
            in_size = sizes[i]
            out_size = sizes[i+1]
            lt = layer_types[i]
            layers.append(instantiate(lt, in_size, out_size, **kwargs))
            if lt == LayerType.NORMAL:
                layers.append(activation)
        self.layers = nn.ModuleList(layers)
       
    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to the network.
            
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """

        self.reset_activations()

        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)