import torch
from torch import nn
import torch.nn.functional as F

from util import deprecation_warning


class LocalHead(nn.Module):
    """
    DEPRECATED  

    A neural network module that applies a linear transformation followed by a custom non-linear transformation.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        old_out_features (int): Number of output features before increment.
        linear (nn.Linear): Linear transformation layer.
        x_min (float): Minimum value for the hardtanh activation.
        x_max (float): Maximum value for the hardtanh activation.
        left_bounds (nn.Parameter): Left bounds for the custom non-linear transformation.
        right_bounds (nn.Parameter): Right bounds for the custom non-linear transformation.

    Methods:
        forward(x):
            Applies the linear transformation, followed by hardtanh activation and custom non-linear transformation.
        increment(new_classes: list[int]):
            Increments the number of output features and updates the linear transformation layer and bounds accordingly.
    """

    def __init__(self, 
        in_features: int,
        initial_out_features: int,
        x_min: float = -1.,
        x_max: float = 1.
    ):
        """
        Initializes the LocalHead layer.

        Args:
            in_features (int): Number of input features.
            initial_out_features (int): Initial number of output features.
            x_min (float, optional): Minimum value of the domain. Defaults to -1.
            x_max (float, optional): Maximum value of the domain. Defaults to 1.

        Raises:
            DeprecationWarning: Indicates that LocalHead is deprecated.
        """
                
        deprecation_warning("LocalHead is deprecated!")

        super().__init__()
        
        self.out_features = initial_out_features
        self.old_out_features = initial_out_features
        self.in_features = in_features

        self.linear = nn.Linear(in_features, initial_out_features)

        self.x_min = x_min
        self.x_max = x_max

        domain = torch.linspace(x_min, x_max, initial_out_features+1)
        left_bounds  = domain[:-1].clone().unsqueeze(0).repeat(initial_out_features,1)
        right_bounds = domain[1:].clone().unsqueeze(0).repeat(initial_out_features,1)
        self.left_bounds  = nn.Parameter(data=left_bounds, requires_grad=False)
        self.right_bounds = nn.Parameter(data=right_bounds, requires_grad=False)


    def forward(self, x):
        """
        Perform the forward pass of the model layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the layer transformations.

        The forward pass includes the following steps:
        1. Apply a linear transformation to the input tensor.
        2. If `out_features` is greater than `old_out_features`, scale and shift the tensor.
        3. Apply the HardTanh activation function with specified min and max values.
        4. Flatten the tensor starting from the second dimension.
        5. Unsqueeze and repeat the tensor along a new dimension.
        6. Compute a normalization constant based on the bounds.
        7. Apply ReLU activation and normalization to the tensor.
        8. Square the tensor and sum along the specified dimension.
        """

        x = self.linear(x)
        if self.out_features > self.old_out_features:
            x *= self.old_out_features / self.out_features
            x -= 1 / (self.x_max - self.x_min)
        x = F.hardtanh(x, self.x_min, self.x_max)

        x = x.flatten(start_dim=1)

        x = x.unsqueeze(2).repeat(1,1,self.out_features)
        norm_const = 4 / (self.right_bounds - self.left_bounds)**2
        
        # Forward pass
        x = torch.relu(x - self.left_bounds) \
            * torch.relu(self.right_bounds - x) \
            * norm_const
        
        x = x * x
        x = x.sum(dim=2)

        return x
    

    @torch.no_grad()
    def increment(self, new_classes: list[int]):
        """
        Increment the output features of the linear layer and update the bounds.
        
        Args:
            new_classes (list[int]): A list of new class indices to be added.

        This method performs the following steps:
        1. Stores the current number of output features.
        2. Increases the number of output features by the length of new_classes.
        3. Creates a new linear layer with the updated number of output features.
        4. Copies the weights and biases from the old linear layer to the new one.
        5. Updates the linear layer to the new one.
        6. Recalculates the domain and bounds based on the new number of output features.
        7. Updates the left and right bounds with the new values.
        """

        old_out_features = self.out_features
        self.out_features += len(new_classes)
        self.old_out_features = old_out_features

        new_linear = nn.Linear(self.in_features, self.out_features)
        new_linear.weight.data[:old_out_features, :] = self.linear.weight.data
        new_linear.bias.data[:old_out_features] = self.linear.bias.data
        self.linear = new_linear

        new_domain = torch.linspace(self.x_min, self.x_max, self.out_features + 1)
        new_left_bounds = new_domain[:-1].unsqueeze(0).repeat(self.out_features, 1)
        new_right_bounds = new_domain[1:].unsqueeze(0).repeat(self.out_features, 1)
        self.left_bounds.data = new_left_bounds
        self.right_bounds.data = new_right_bounds