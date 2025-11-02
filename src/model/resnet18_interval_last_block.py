import torch
import torch.nn as nn

from .cl_module_abc import CLModuleABC
from .layer.interval_activation import IntervalActivation
from .inc_classifier import IncrementalClassifier

from typing import Tuple, Union

class FlattenedResNet18FE(nn.Module):
    """A flattened ResNet-18 feature extractor."""
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            original_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except (ImportError, Exception):
            print("Could not load pretrained weights. Initializing with random weights.")
            from torchvision.models import resnet18
            original_model = resnet18(weights=None)

        # Feature Extractor Layers
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        # Layer 1
        self.layer1_0_conv1 = original_model.layer1[0].conv1; self.layer1_0_bn1 = original_model.layer1[0].bn1
        self.layer1_0_conv2 = original_model.layer1[0].conv2; self.layer1_0_bn2 = original_model.layer1[0].bn2
        self.layer1_1_conv1 = original_model.layer1[1].conv1; self.layer1_1_bn1 = original_model.layer1[1].bn1
        self.layer1_1_conv2 = original_model.layer1[1].conv2; self.layer1_1_bn2 = original_model.layer1[1].bn2
        # Layer 2
        self.layer2_0_conv1 = original_model.layer2[0].conv1; self.layer2_0_bn1 = original_model.layer2[0].bn1
        self.layer2_0_conv2 = original_model.layer2[0].conv2; self.layer2_0_bn2 = original_model.layer2[0].bn2
        self.layer2_0_downsample = original_model.layer2[0].downsample
        self.layer2_1_conv1 = original_model.layer2[1].conv1; self.layer2_1_bn1 = original_model.layer2[1].bn1
        self.layer2_1_conv2 = original_model.layer2[1].conv2; self.layer2_1_bn2 = original_model.layer2[1].bn2
        # Layer 3
        self.layer3_0_conv1 = original_model.layer3[0].conv1; self.layer3_0_bn1 = original_model.layer3[0].bn1
        self.layer3_0_conv2 = original_model.layer3[0].conv2; self.layer3_0_bn2 = original_model.layer3[0].bn2
        self.layer3_0_downsample = original_model.layer3[0].downsample
        self.layer3_1_conv1 = original_model.layer3[1].conv1; self.layer3_1_bn1 = original_model.layer3[1].bn1
        self.layer3_1_conv2 = original_model.layer3[1].conv2; self.layer3_1_bn2 = original_model.layer3[1].bn2
        # Layer 4
        self.layer4_0_conv1 = original_model.layer4[0].conv1; self.layer4_0_bn1 = original_model.layer4[0].bn1
        self.layer4_0_conv2 = original_model.layer4[0].conv2; self.layer4_0_bn2 = original_model.layer4[0].bn2
        self.layer4_0_downsample = original_model.layer4[0].downsample
        self.layer4_1_conv1 = original_model.layer4[1].conv1; self.layer4_1_bn1 = original_model.layer4[1].bn1
        self.layer4_1_conv2 = original_model.layer4[1].conv2; self.layer4_1_bn2 = original_model.layer4[1].bn2
        
        self.avgpool = original_model.avgpool
        self.final_relu = nn.ReLU(inplace=True)
        self.relu_int = nn.ReLU(inplace=True)

class ResNet18IntervalLastBlock(CLModuleABC):
    """
    A flattened ResNet-18 backbone augmented with IntervalActivation layers.
    
    This implementation uses a non-nested ResNet-18 structure, allowing for
    the direct insertion of IntervalActivation layers within layer4 and,
    optionally, between the residual blocks of layer4.
    """
    def __init__(
        self,
        initial_out_features: int,
        dim_hidden: int,
        interval_layer_kwargs: dict = None,
        head_type: str = "Normal",
        mask_past_classifier_neurons: bool = False,
        head_kwargs: dict = {},
    ) -> None:
        head = IncrementalClassifier(
            in_features=dim_hidden,
            initial_out_features=initial_out_features,
            head_type=head_type,
            mask_past_classifier_neurons=mask_past_classifier_neurons,
            **head_kwargs,
        )
        super().__init__(head)

        if interval_layer_kwargs is None:
            interval_layer_kwargs = {"lower_percentile": 0.05, "upper_percentile": 0.95}

        self.fe = FlattenedResNet18FE()
        self._insert_interval_activations(interval_layer_kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(512, dim_hidden),
            IntervalActivation(layer_name="interval_head",**interval_layer_kwargs),
        )

        self.freeze_backbone()

    def _insert_interval_activations(self, kwargs: dict) -> None:
        """Dynamically adds IntervalActivation layers for use in the forward pass."""
        # Intervals for the first block of layer4
        self.interval_l4_0_conv1 = IntervalActivation(layer_name="interval_l4_0_conv1", **kwargs)
        self.interval_l4_0_bn1 = IntervalActivation(layer_name="interval_l4_0_bn1", **kwargs)
        self.interval_l4_0_conv2 = IntervalActivation(layer_name="interval_l4_0_conv2", **kwargs)
        self.interval_l4_0_bn2 = IntervalActivation(layer_name="interval_l4_0_bn2", **kwargs)

        self.interval_l4_0_downsample_0 = IntervalActivation(layer_name="interval_l4_0_downsample_0", **kwargs)
        self.interval_l4_0_downsample_1 = IntervalActivation(layer_name="interval_l4_0_downsample_1", **kwargs)
        
        # Intervals for the second block of layer4
        self.interval_l4_1_conv1 = IntervalActivation(layer_name="interval_l4_1_conv1", **kwargs)
        self.interval_l4_1_bn1 = IntervalActivation(layer_name="interval_l4_1_bn1", **kwargs)
        self.interval_l4_1_conv2 = IntervalActivation(layer_name="interval_l4_1_conv2", **kwargs)
        self.interval_l4_1_bn2 = IntervalActivation(layer_name="interval_l4_1_bn2", **kwargs)


    def freeze_backbone(self) -> None:
        """Freezes all parameters except those in layer4 and interval layers."""
        for name, param in self.fe.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False


    def forward_features(
        self,
        x: torch.Tensor,
        return_first_interval_activation: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the flattened feature extractor.
        If `return_first_interval_activation=True`, returns a tuple:
            (features, first_interval_activation)
        """

        # --- Standard ResNet Layers ---
        x = self.fe.conv1(x); x = self.fe.bn1(x); x = self.fe.relu(x); x = self.fe.maxpool(x)
        # Layer 1
        identity = x
        out = self.fe.layer1_0_conv1(x); out = self.fe.layer1_0_bn1(out); out = self.fe.relu_int(out)
        out = self.fe.layer1_0_conv2(out); out = self.fe.layer1_0_bn2(out)
        out += identity; out = self.fe.final_relu(out)
        identity = out
        out = self.fe.layer1_1_conv1(out); out = self.fe.layer1_1_bn1(out); out = self.fe.relu_int(out)
        out = self.fe.layer1_1_conv2(out); out = self.fe.layer1_1_bn2(out)
        out += identity; x = self.fe.final_relu(out)
        # Layer 2
        identity = self.fe.layer2_0_downsample(x)
        out = self.fe.layer2_0_conv1(x); out = self.fe.layer2_0_bn1(out); out = self.fe.relu_int(out)
        out = self.fe.layer2_0_conv2(out); out = self.fe.layer2_0_bn2(out)
        out += identity; out = self.fe.final_relu(out)
        identity = out
        out = self.fe.layer2_1_conv1(out); out = self.fe.layer2_1_bn1(out); out = self.fe.relu_int(out)
        out = self.fe.layer2_1_conv2(out); out = self.fe.layer2_1_bn2(out)
        out += identity; x = self.fe.final_relu(out)
        # Layer 3
        identity = self.fe.layer3_0_downsample(x)
        out = self.fe.layer3_0_conv1(x); out = self.fe.layer3_0_bn1(out); out = self.fe.relu_int(out)
        out = self.fe.layer3_0_conv2(out); out = self.fe.layer3_0_bn2(out)
        out += identity; out = self.fe.final_relu(out)
        identity = out
        out = self.fe.layer3_1_conv1(out); out = self.fe.layer3_1_bn1(out); out = self.fe.relu_int(out)
        out = self.fe.layer3_1_conv2(out); out = self.fe.layer3_1_bn2(out)
        out += identity; x = self.fe.final_relu(out)

        # --- Layer 4 with Interval Activations ---
        # Block 4.0
        identity = self.fe.layer4_0_downsample[0](x)
        identity = self.interval_l4_0_downsample_0(identity)
        identity_first_interval_activation = identity.clone()

        identity = self.fe.layer4_0_downsample[1](identity)
        identity = self.interval_l4_0_downsample_1(identity)

        out = self.fe.layer4_0_conv1(x)
        out = self.interval_l4_0_conv1(out)
        layer4_0_first_interval_activation = out.clone()

        out = self.fe.layer4_0_bn1(out)
        out = self.interval_l4_0_bn1(out)

        out = self.fe.layer4_0_conv2(out)
        out = self.interval_l4_0_conv2(out)

        out = self.fe.layer4_0_bn2(out)
        out = self.interval_l4_0_bn2(out)

        out += identity
        out = self.fe.final_relu(out)
        
        # Block 4.1
        identity = out
        out = self.fe.layer4_1_conv1(out)
        out = self.interval_l4_1_conv1(out)

        out = self.fe.layer4_1_bn1(out)
        out = self.interval_l4_1_bn1(out)

        out = self.fe.layer4_1_conv2(out)
        out = self.interval_l4_1_conv2(out)

        out = self.fe.layer4_1_bn2(out)
        out += identity
        x = self.fe.final_relu(out)
    
        # --- Final Pooling ---
        x = self.fe.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.interval_l4_1_bn2(x)

        if return_first_interval_activation:
            return x, identity_first_interval_activation, layer4_0_first_interval_activation
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_first_interval_activation: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.forward_features(x, return_first_interval_activation=return_first_interval_activation)
        if return_first_interval_activation:
            _, identity_first_interval_activation, layer4_0_first_interval_activation = feats
        
            return identity_first_interval_activation, layer4_0_first_interval_activation
        else:
            feats = self.mlp(feats)
            return self.head(feats)