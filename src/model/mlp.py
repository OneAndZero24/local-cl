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
        layer_types: list[str]=["Normal"],
        **kwargs
    ):
        head_type = LayerType(head_type)
        layer_types = map(lambda x: LayerType(x), layer_types)

        head_kwargs = kwargs.copy()
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
            for lt in layer_types:
                if layers:
                    in_size = sizes[i+1]
                    out_size = sizes[i+1]

                if lt == LayerType.NORMAL:
                    if layers and (type(layers[-1]) == nn.Linear):
                        layers.append(nn.Tanh())
                    layers.append(nn.Linear(in_size, out_size))
                else:
                    layers.append(LocalLayer(in_size, out_size, **kwargs))
                    
            if layers and (type(layers[-1]) == nn.Linear):
                layers.append(nn.Tanh())
        self.layers = nn.ModuleList(layers)
       
    def forward(self, x):
        self.reset_activations()

        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)