from abc import ABCMeta

from torch import nn

from model.layer import LocalLayer, LocalConv2DLayer


LOCAL_LAYERS = (LocalLayer, LocalConv2DLayer)

class ActivationRecordingModuleABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, head: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = None
        self.head = head


    def reset_activations(self):
        self.activations = []


    def add_activation(self, layer: nn.Module, x):
        if isinstance(layer, LOCAL_LAYERS):
            self.activations.append(x)