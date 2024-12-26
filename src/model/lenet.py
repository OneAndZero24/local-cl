import torch.nn as nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer import LayerType, LocalLayer, instantiate2D


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
        head_type: LayerType=LayerType.NORMAL,
        add_avg_pool: bool=True,
        add_fc_local: bool=False,
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
            nn.Linear(sizes[-1], head_size),
            LocalLayer(head_size, head_size, **kwargs) if add_fc_local else nn.Identity,
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
