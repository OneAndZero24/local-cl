from abc import ABCMeta

from torch import nn
import torch

from model.layer import LocalLayer, LocalConv2DLayer


LOCAL_LAYERS = (LocalLayer, LocalConv2DLayer)
AFFINE_LAYERS = (nn.Linear, nn.Conv2d)

class ActivationRecordingModuleABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, head: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = None
        self.active_hills = None
        self.head = head


    def reset_activations(self):
        self.activations = []


    def reset_active_hills(self):
        self.active_hills = []


    def add_activation(self, layer: nn.Module, x):
        if isinstance(layer, LOCAL_LAYERS):
            self.activations.append(x)

    @torch.no_grad()
    def find_active_hills(self, curr_layer: nn.Module, next_layer: nn.Module, x):
        if isinstance(curr_layer, AFFINE_LAYERS) and isinstance(next_layer, LOCAL_LAYERS):
            x = x.clone()
            left_bounds = next_layer.left_bounds.clone()
            right_bounds = next_layer.right_bounds.clone()

            x = x.unsqueeze(2).repeat(1, 1, next_layer.out_features)
            left_bounds = left_bounds.unsqueeze(0).repeat(x.shape[0], 1, 1)
            right_bounds = right_bounds.unsqueeze(0).repeat(x.shape[0], 1, 1)

            active_condition = (x >= left_bounds) & (x < right_bounds)
            active_condition = active_condition.int()

            self.active_hills.append(active_condition)