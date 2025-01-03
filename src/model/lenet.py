import torch.nn as nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer import LayerType, instantiate2D


class LeNet(ActivationRecordingModuleABC):
    """
    LeNet model with incremental classifier head.
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
        self.reset_activations()

        for layer in self.conv_layers:
            x = layer(x)
            self.add_activation(layer, x)
        
        for layer in self.fc_layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)
