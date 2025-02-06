import torch.nn as nn

from util import deprecation_warning
from model.activation_recording_abc import ActivationRecordingModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer import LayerType, instantiate2D


class LeNet(ActivationRecordingModuleABC):
    """
    DEPRECATED  

    LeNet is a convolutional neural network model that extends the ActivationRecordingModuleABC.
    This implementation includes convolutional layers, activation functions, and fully connected layers.

    Attributes:
        conv_layers (nn.ModuleList): List of convolutional layers with activation functions and optional average pooling.
        fc_layers (nn.ModuleList): List of fully connected layers with activation functions.

    Methods:
        forward(x):
            Defines the forward pass of the network. It processes the input through convolutional layers,
            records activations, and then processes through fully connected layers before passing to the head.

    Args:
        size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolutional kernel.
        initial_out_features (int): Number of output features from the initial layer.
        sizes (list[int]): List of sizes for each layer.
        head_size (int): Size of the head layer.
        conv_type (str, optional): Type of convolutional layer to use. Defaults to "Normal".
        head_type (str, optional): Type of head layer to use. Defaults to "Normal".
        add_avg_pool (bool, optional): Whether to add average pooling layers. Defaults to True.
        **kwargs: Additional keyword arguments for layer instantiation.
    """


    def __init__(self,
        size: int,
        stride: int,
        initial_out_features: int, 
        sizes: list[int],
        head_size: int,
        conv_type: str="Normal",
        head_type: str="Normal",
        add_avg_pool: bool=True,
        **kwargs
    ):
        """
        Initializes the LeNet model.

        Args:
            size (int): The size of the convolutional kernel.
            stride (int): The stride of the convolutional kernel.
            initial_out_features (int): The number of initial output features.
            sizes (list[int]): A list of sizes for each layer.
            head_size (int): The size of the head layer.
            conv_type (str, optional): The type of convolutional layer. Defaults to "Normal".
            head_type (str, optional): The type of head layer. Defaults to "Normal".
            add_avg_pool (bool, optional): Whether to add average pooling layers. Defaults to True.
            **kwargs: Additional keyword arguments.

        Raises:
            DeprecationWarning: Indicates that the LeNet model is deprecated.
        """
                
        deprecation_warning("LeNet is deprecated!")

        conv_type = LayerType(conv_type)
        head_type = LayerType(head_type)
        super().__init__(
            IncrementalClassifier(
                head_size, 
                initial_out_features,
                head_type,
                **kwargs
            )
        )

        kwargs.pop("masking", None)
        kwargs.pop("mask_value", None)

        layers = []
        for i in range(len(sizes)-1):
            layers.append(instantiate2D(conv_type, sizes[i], sizes[i+1], size, stride, **kwargs))
            layers.append(nn.Tanh())
            if add_avg_pool and (i < len(sizes)-2):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                layers.append(nn.Tanh())

        self.conv_layers = nn.ModuleList(layers)

        layers = [
            nn.Flatten(),
            nn.Linear(sizes[-1], head_size),
            nn.Tanh()
        ]
        self.fc_layers = nn.ModuleList(layers)


    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor to the model.
            
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """

        self.reset_activations()

        for layer in self.conv_layers:
            x = layer(x)
            self.add_activation(layer, x)
        
        for layer in self.fc_layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)
