import torch
import torch.nn as nn

from model.activation_recording_abc import ActivationRecordingModuleABC
from model.inc_classifier import IncrementalClassifier
from model.layer import LayerType, LocalLayer


class MLP(ActivationRecordingModuleABC):
    """
    Simple MLP model with incremental classifier head.
    """

    def __init__(self,
        initial_out_features: int,
        sizes: list[int],
        head_type: str="Normal",
        add_fc_local: bool=True,
        **kwargs
    ):
        head_type = LayerType(head_type)
        super().__init__(
            IncrementalClassifier(
                sizes[-1], 
                initial_out_features,
                head_type,
                **kwargs
            )
        )

        kwargs.pop("masking", None)
        kwargs.pop("mask_value", None)

        layers = []
        N = len(sizes)-1
        for i in range(N):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if add_fc_local and (i < N): # dont add Local after last
                layers.append(LocalLayer(sizes[i+1], sizes[i+1], **kwargs))
            else:
                layers.append(nn.Tanh())
        self.layers = nn.ModuleList(layers)
       
    def forward(self, x):
        self.reset_activations()

        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)