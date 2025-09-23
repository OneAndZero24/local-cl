import torch
from torch import nn

from model.layer import instantiate, LayerType, LocalModule


class IncrementalClassifier(nn.Module):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.

    Attributes:
        masking (bool): Whether to apply masking to the output.
        mask_value (int): The value to use for masked outputs.
        mul (float): A multiplier used during forward pass when old classes exist.
        old_nclasses (int or None): The number of classes before the last increment.
        get_classifier (Callable): A function to instantiate the classifier layer.
        classifier (nn.Module): The classifier layer.
        active_units (torch.Tensor): A tensor indicating active units for masking.

    Methods:
        increment(new_classes: list[int]):
            Increment the classifier to accommodate new classes.
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the classifier.
    """


    def __init__(
        self,
        in_features: int,
        initial_out_features: int=2,
        layer_type: LayerType=LayerType.NORMAL,
        masking: bool=False,
        mask_value: int=-1000,
        mask_past_classifier_neurons: bool = False,
        **kwargs,
    ):
        """
        Initializes the IncClassifier.

        Args:
            in_features (int): Number of input features.
            initial_out_features (int, optional): Number of initial output features. Defaults to 2.
            layer_type (LayerType, optional): Type of layer to use for the classifier. Defaults to LayerType.NORMAL.
            masking (bool, optional): Whether to apply masking. Defaults to True.
            mask_value (int, optional): Value to use for masking. Defaults to -1000.
            **kwargs: Additional keyword arguments to pass to the layer instantiation.

        Attributes:
            masking (bool): Whether masking is enabled.
            mask_value (int): Value used for masking.
            mul (float): Multiplier value, initialized to 1.0.
            old_nclasses (None): Placeholder for old number of classes, initialized to None.
            last_logits (None): Stores the last classifier output.
            get_classifier (function): Lambda function to instantiate the classifier layer.
            classifier (nn.Module): The instantiated classifier layer.
            active_units (torch.Tensor): Buffer to keep track of active units, initialized to zeros.
        """

        super().__init__()
        self.masking = masking
        self.mask_value = mask_value
        self.mask_past_classifier_neurons = mask_past_classifier_neurons

        self.old_nclasses = None
        self.last_logits = None

        self.get_classifier = (lambda in_features, out_features: 
            instantiate(
                layer_type,
                in_features, 
                out_features,
                **kwargs
            )
        )

        self.classifier = self.get_classifier(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)


    @torch.no_grad()
    def increment(self, new_classes: list[int]):
        """
        Increment the classifier to accommodate new classes.

        Args:
            new_classes (list[int]): A list of new class indices to be added.

        This method updates the classifier to handle new classes by adjusting the 
        output layer and active units if masking is enabled. It ensures that the 
        classifier's parameters are updated to reflect the new number of classes.
        Steps:
        1. Determine the device on which the classifier's parameters are located.
        2. Calculate the new number of classes based on the maximum index in new_classes.
        3. If masking is enabled and the number of classes has changed:
            - Update the active units tensor to accommodate the new classes.
        4. If the number of classes has changed:
            - Adjust the classifier's parameters to reflect the new number of classes.
            - Update the classifier's state dictionary with the new parameters.
        """

        device = next(self.classifier.parameters()).device

        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = max(old_nclasses, max(new_classes)+1)

        if self.masking:
            if old_nclasses != new_nclasses: 
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[:old_act_units.shape[0]] = old_act_units

            self.active_units[new_classes] = 1

        if old_nclasses != new_nclasses:
            self.old_nclasses = old_nclasses
            state_dict = self.classifier.state_dict()
            self.classifier = self.get_classifier(in_features, new_nclasses).to(device)
            param_filter = []
            idx = slice(None, old_nclasses)
            if isinstance(self.classifier, LocalModule):
                param_filter = self.classifier.incrementable_params()
                idx = self.classifier.get_slice(old_nclasses)
            if isinstance(self.classifier, nn.Linear):
                param_filter.extend(["weight", "bias"])
            for name, param in self.classifier.named_parameters():
                if (name in param_filter):
                        param.data[idx] = state_dict[name]
                else:
                    param.data = state_dict[name]

    def forward(self, x):
        """
        Perform a forward pass through the classifier.

        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying the classifier and optional masking.
        The forward pass includes the following steps:
        1. Pass the (possibly modified) input tensor through the classifier.
        2. If masking is enabled and the model is in training mode:
            - Create a mask from the active_units tensor.
            - Apply the mask to the output tensor, filling masked positions with mask_value.
        """
        out = self.classifier(x)
        if self.masking and self.training:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)

        if self.mask_past_classifier_neurons and self.old_nclasses is not None and self.training:
            out[:, :self.old_nclasses] = -float('inf')
        self.last_logits = out

        return out