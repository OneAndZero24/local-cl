import torch
import torch.nn as nn

from activation_recording_abc import ActivationRecordingModuleABC
from inc_classifier import IncrementalClassifier
from layer import LayerType, LocalLayer


class MLP(ActivationRecordingModuleABC):
    """
    Simple MLP model with incremental classifier head.
    """

    def __init__(self,
        initial_out_features: int,
        sizes: list[int],
        head_type: LayerType=LayerType.NORMAL,
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
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(LocalLayer(sizes[i+1], sizes[i+1], **kwargs))

        self.layers = nn.ModuleList(layers)
    
    
    def forward(self, x):
        self.reset_activations()

        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)
