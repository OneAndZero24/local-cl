import torch
from torch import nn

from model import instantiate, LayerType


class IncrementalClassifier(nn.Module):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(
        self,
        in_features: int,
        initial_out_features: int=2,
        layer_type: LayerType=LayerType.NORMAL,
        masking: bool=True,
        mask_value: int=-1000,
        **kwargs,
    ):
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value

        self.get_classifier = (lambda in_features, out_features: 
            instantiate(
                in_features, 
                out_features,
                layer_type,
                **kwargs
            )
        )

        self.classifier = self.get_classifier(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)


    @torch.no_grad()
    def increment(self, new_classes: list[int]):
        device = next(self.classifier.parameters()).device

        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = len(new_classes)
        new_nclasses = max(old_nclasses, new_nclasses)

        if self.masking:
            if old_nclasses != new_nclasses: 
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[:old_act_units.shape[0]] = old_act_units

            if self.training:
                self.active_units[new_classes] = 1

        if old_nclasses != new_nclasses:
            state_dict = self.classifier.state_dict()
            self.classifier = self.get_classifier(in_features, new_nclasses).to(device)
            for name, param in self.classifier.named_parameters():
                param.data[:old_nclasses] = state_dict[name]


    def forward(self, x):
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)
        return out