import torch

from avalanche.benchmarks.scenarios import CLExperience
from avalanche.models.dynamic_modules import DynamicModule

from models import instantiate


class IncrementalClassifier(DynamicModule):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        layer_type: str = "Linear",
        masking=True,
        mask_value=-1000,
        **kwargs,
    ):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__(**kwargs)
        self.masking = masking
        self.mask_value = mask_value

        train_domain = kwargs["train_domain"] if "train_domain" in kwargs else True
        toggle_linear = kwargs["toggle_linear"] if "toggle_linear" in kwargs else False

        self.get_classifier = (lambda in_features, out_features: 
            instantiate(in_features, 
                        out_features,
                        layer_type,
                        train_domain,
                        toggle_linear
            )
        )

        self.classifier = get_classifier(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        device = self._adaptation_device
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[list(curr_classes)] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        state_dict = self.classifier.state_dict()
        self.classifier = self.get_classifier(in_features, new_nclasses).to(device)
        for name, param in self.classifier.named_parameters():
            param.data[:old_nclasses] = state_dict["name"]

    def forward(self, x, **kwargs):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)
        return out