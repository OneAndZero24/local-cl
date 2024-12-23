import torch.nn as nn

from activation_recording_abc import ActivationRecordingModuleABC
from inc_classifier import IncrementalClassifier
from layer import LayerType, instantiate2D, instantiate


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
        conv_type: LayerType=LayerType.NORMAL,
        fc_type: LayerType=LayerType.NORMAL,
        head_type: LayerType=LayerType.NORMAL,
        add_avg_pool: bool=True,
        **kwargs
    ):
        masking = kwargs["masking"]
        mask_value = kwargs["mask_value"]
        del kwargs["masking"]
        del kwargs["mask_value"]
        super().__init__(
            IncrementalClassifier(
                sizes[-1], 
                initial_out_features,
                layer_type=head_type,
                masking=masking,
                mask_value=mask_value
                **kwargs
            )
        )

        layers = []
        for i in range(len(sizes)-1):
            layers.append(instantiate2D(conv_type, sizes[i], sizes[i+1], size, stride, **kwargs))
            layers.append(nn.Tanh())
            if add_avg_pool and (i < len(sizes)-2):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                layers.append(nn.Tanh())

        self.conv_layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            instantiate(fc_type, sizes[-1], head_size, head_type, **kwargs),
            nn.Tanh()
        )


    def forward(self, x):
        self.reset_activations()

        for layer in self.conv_seq:
            x = layer(x)
            self.add_activation(layer, x)
        
        for layer in self.fc_seq:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)
