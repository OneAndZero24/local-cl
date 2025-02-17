from abc import ABCMeta

from torch import nn
from model.layer import LocalModule


class CLModuleABC(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for modules that record activations of specified layers during forward passes.

    Attributes:
        activations (list): A list to store activations recorded from specified layers.
        head (nn.Module): The head module of the neural network.

    Methods:
        reset_activations():
            Resets the activations list to an empty list.
        add_activation(layer: nn.Module, x):
            Adds the activation `x` to the activations list if the layer is an instance of LOCAL_LAYERS.
    """

    def __init__(self, head: nn.Module, *args, **kwargs):
        """
        Initializes the activation recording module.

        Args:
            head (nn.Module): The neural network module whose activations are to be recorded.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self.activations = None
        self.head = head


    def reset_activations(self):
        """
        Resets the activations list to an empty list.
        This method clears the current activations recorded in the model by
        setting the `activations` attribute to an empty list.
        """

        del self.activations
        self.activations = []


    def add_activation(self, layer: nn.Module, x):
        """
        Adds the activation output of a given layer to the activations list.

        Args:
            layer (nn.Module): The neural network layer whose activation is to be recorded.
            x: The activation output of the layer.
            
        Returns:
            None
        """
        
        self.activations.append(x)