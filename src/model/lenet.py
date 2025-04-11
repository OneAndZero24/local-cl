from typing import Union
from omegaconf import ListConfig

import torch.nn as nn

from model.cl_module_abc import CLModuleABC
from model.mlp import MLP
from model.layer import LayerType, instantiate2D


class LeNet(CLModuleABC):
    """
    LeNet is a convolutional neural network module that combines convolutional layers 
    and a multi-layer perceptron (MLP) for feature extraction and classification.

    Args:
        initial_out_features (int): The number of output features from the convolutional layers 
            that will be passed to the MLP.
        kernel_sizes (list[int]): A list of kernel sizes for each convolutional layer.
        kernel_strides (list[int]): A list of strides for each convolutional layer.
        sizes (list[int]): A list of input and output channel sizes for each convolutional layer. 
            The length of this list should be one more than the length of `kernel_sizes`.
        mlp_sizes (list[int]): A list of sizes for the MLP layers.
        mlp_layers (list[str]): A list of layer types for the MLP.
        mlp_activation (nn.Module, optional): The activation function to use in the MLP. 
            Defaults to `nn.Tanh()`.
        conv_activation (nn.Module, optional): The activation function to use in the convolutional 
            layers. Defaults to `nn.Tanh()`.
        mlp_config (Union[dict, list[dict], ListConfig], optional): Configuration for the MLP layers. 
            Defaults to an empty dictionary.
        conv_config (Union[dict, list[dict], ListConfig], optional): Configuration for the convolutional 
            layers. Defaults to an empty dictionary.
        head_type (str, optional): The type of head to use in the MLP. Defaults to `"Normal"`.
        conv_type (str, optional): The type of convolutional layer to use. Defaults to `"Normal"`.
        add_avg_pool (bool, optional): Whether to add average pooling layers after each convolutional 
            layer except the last one. Defaults to `True`.

    Attributes:
        conv_layers (nn.ModuleList): A list of convolutional layers, including activation functions 
            and optional average pooling layers.
        mlp (MLP): The multi-layer perceptron module used for classification.
    
    Methods:
        forward(x):
            Performs a forward pass through the network. Applies the convolutional layers to the input, 
            followed by the MLP for classification.

    Raises:
        AssertionError: If the lengths of `kernel_sizes` and `kernel_strides` do not match.
        AssertionError: If the length of `kernel_sizes` does not match `len(sizes) - 1`.
    """
    
    def __init__(self,
        initial_out_features: int,
        kernel_sizes: list[int],
        kernel_strides: list[int], 
        sizes: list[int],
        mlp_sizes: list[int],
        mlp_layers: list[str],
        mlp_activation: nn.Module=nn.Tanh(),
        conv_activation: nn.Module=nn.Tanh(),
        mlp_config: Union[dict, list[dict], ListConfig]={},
        conv_config: Union[dict, list[dict], ListConfig]={},
        head_type: str="Normal",
        conv_type: str="Normal",
        add_avg_pool: bool=True
    ):
        """
        Initializes the model with convolutional and MLP layers.

        Args:
            initial_out_features (int): The number of output features for the MLP.
            kernel_sizes (list[int]): List of kernel sizes for the convolutional layers.
            kernel_strides (list[int]): List of strides for the convolutional layers.
            sizes (list[int]): List of sizes for the convolutional layers, including input and output sizes.
            mlp_sizes (list[int]): List of sizes for the MLP layers.
            mlp_layers (list[str]): List of layer types for the MLP.
            mlp_activation (nn.Module, optional): Activation function for the MLP layers. Defaults to nn.Tanh().
            conv_activation (nn.Module, optional): Activation function for the convolutional layers. Defaults to nn.Tanh().
            mlp_config (Union[dict, list[dict], ListConfig], optional): Configuration for the MLP layers. Defaults to {}.
            conv_config (Union[dict, list[dict], ListConfig], optional): Configuration for the convolutional layers. Defaults to {}.
            head_type (str, optional): Type of the head layer for the MLP. Defaults to "Normal".
            conv_type (str, optional): Type of the convolutional layers. Defaults to "Normal".
            add_avg_pool (bool, optional): Whether to add average pooling layers after convolutional layers. Defaults to True.

        Raises:
            AssertionError: If the lengths of `kernel_sizes` and `kernel_strides` do not match.
            AssertionError: If the length of `kernel_sizes` does not match `len(sizes) - 1`.
        """

        assert len(kernel_sizes) == len(kernel_strides), "Kernel sizes and strides must match"
        assert len(kernel_sizes) == len(sizes)-1, "Kernel sizes and sizes must match"

        mlp = MLP(
            initial_out_features,
            mlp_sizes,
            mlp_layers,
            head_type,
            mlp_activation,
            mlp_config
        )
        super().__init__(mlp.head)

        self.mlp = mlp

        conv_type = LayerType(conv_type)

        config = conv_config
        list_config = isinstance(config, (list, ListConfig))

        layers = []
        for i in range(len(sizes)-1):
            layers.append(
                instantiate2D(
                    conv_type, sizes[i], sizes[i+1], kernel_sizes[i], kernel_strides[i], 
                    **((config[i] if i < len(config) else config[-1]) if list_config else config)
                )
            )
            layers.append(conv_activation)
            if add_avg_pool and (i < len(sizes)-2):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv_layers = nn.ModuleList(layers)
        self.layers = self.conv_layers+self.mlp.layers


    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional
                          layers and the MLP (multi-layer perceptron).

        Notes:
            - Resets activations at the start of the forward pass.
            - Iteratively applies each convolutional layer to the input tensor
              and stores the activations.
        """

        self.reset_activations()

        for layer in self.conv_layers:
            x = layer(x)
            self.add_activation(layer, x)

        return self.mlp(x)
