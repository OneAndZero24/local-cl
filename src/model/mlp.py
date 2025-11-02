from typing import Union
from omegaconf import ListConfig

import torch
import torch.nn as nn

from model.cl_module_abc import CLModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer import instantiate, LayerType
from model.layer.interval_activation import IntervalActivation


class MLP(CLModuleABC):
    """
    Multi-Layer Perceptron (MLP) class that extends CLModuleABC.

    Args:
        initial_out_features (int): The number of output features for the initial layer.
        sizes (list[int]): A list of integers representing the sizes of each layer.
        mlp (list[str]): A list of strings representing the type of each layer.
        head_type (str, optional): The type of the head layer. Defaults to "Normal".
        activation (nn.Module, optional): The activation function to use between layers. Defaults to nn.Tanh().
        **kwargs: Additional keyword arguments for layer instantiation.

    Attributes:
        mlp (nn.ModuleList): A list of instantiated layers.

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
        mask_past_classifier_neurons: bool = False,
        config: Union[dict, list[dict], ListConfig]={},
    ):
        """
        Initializes the MLP model.
        
        Args:
            initial_out_features (int): The initial number of output features.
            sizes (list[int]): A list of integers representing the sizes of each layer.
            layers (list[str]): A list of strings representing the types of each layer.
            head_type (str, optional): The type of the head layer. Defaults to "Normal".
            activation (nn.Module, optional): The activation function to use between layers. Defaults to nn.Tanh().
            config (Union[dict, list[dict]]): Additional keyword arguments for layer instantiation. 
            If dict passed will get applied to each layer, if list of dicts will apply to each layer in order, head=0. 
            If length of dict shorter than number of layers will use last dict for remaining layers.

        Keyword Args:
            train_head_domain (bool, optional): If head_type is LOCAL, specifies whether to train the head domain. Defaults to False.
            masking (optional): Masking parameter for layers.
            mask_value (optional): Mask value parameter for layers.
        """
                
        assert len(sizes)-1 == len(layers), "Number of sizes and layers must match"
        
        activation = globals()[activation] if isinstance(activation, str) else activation
        
        head_type = LayerType(head_type)
        layer_types = list(map(lambda x: LayerType(x), layers))

        list_config = isinstance(config, (list, ListConfig))

        head_kwargs = config
        if list_config:
            head_kwargs = config[0]
            config = config[1:]

        super().__init__(
            IncrementalClassifier(
                sizes[-1], 
                initial_out_features,
                head_type,
                mask_past_classifier_neurons=mask_past_classifier_neurons,
                **head_kwargs
            )
        )

        layers = []
        N = len(sizes)-1
        for i in range(N):                    
            in_size = sizes[i]
            out_size = sizes[i+1]
            lt = layer_types[i]
            layer_cfg = (config[i] if i < len(config) else config[-1]) if list_config else config
            layers.append(instantiate(lt, in_size, out_size, **layer_cfg))
            try:
                layers.append(activation())
            except TypeError:
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

        return self.head(x)