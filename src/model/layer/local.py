import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_module import LocalModule


class LocalLayer(LocalModule):
    """
    A custom neural network layer that applies a local transformation to the input features.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        train_domain (bool, optional): Whether the domain bounds are trainable. Default is True.
        x_min (float, optional): Minimum value of the input domain. Default is -1.
        x_max (float, optional): Maximum value of the input domain. Default is 1.
        device (torch.device, optional): The device on which to create the parameters. Default is None.
        dtype (torch.dtype, optional): The data type of the parameters. Default is None.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-8.
        use_importance_params (bool, optional): Whether to use importance parameters. Default is True.

    Attributes:
        use_importance_params (bool): Whether to use importance parameters.
        eps (float): A small value to avoid division by zero.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        train_domain (bool): Whether the domain bounds are trainable.
        x_min (float): Minimum value of the input domain.
        x_max (float): Maximum value of the input domain.
        left_bounds (torch.nn.Parameter): Left bounds of the domain.
        right_bounds (torch.nn.Parameter): Right bounds of the domain.
        w (torch.nn.Parameter, optional): Importance parameters.

    Methods:
        reset_parameters():
            Initializes the parameters of the layer.
        forward(x):
            Applies the layer transformation to the input tensor x.
        extra_repr():
            Returns a string with the extra representation of the layer.
    """


    def __init__(self, 
        in_features: int,
        out_features: int,
        train_domain: bool = True,
        x_min: float = -1.,
        x_max: float = 1.,
        device = None, 
        dtype = None,
        eps: float = 1e-8,
        use_importance_params: bool = True
    ):
        """
        Initializes the LocalLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            train_domain (bool, optional): Whether the domain is trainable. Defaults to True.
            x_min (float, optional): Minimum value for x. Defaults to -1.
            x_max (float, optional): Maximum value for x. Defaults to 1.
            device (optional): Device on which the tensor is allocated. Defaults to None.
            dtype (optional): Data type of the tensor. Defaults to None.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.
            use_importance_params (bool, optional): Whether to use importance parameters. Defaults to True.
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_importance_params = use_importance_params

        self.eps = eps if train_domain else 0.0

        self.in_features = in_features
        self.out_features = out_features
        self.train_domain = train_domain

        self.x_min = x_min
        self.x_max = x_max

        self.left_bounds = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs), requires_grad=train_domain)
        self.right_bounds = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs), requires_grad=train_domain)
        if use_importance_params:
            self.w = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs), requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        """
        Resets the parameters of the layer.
        This method initializes the left and right bounds of the layer using a 
        linear space between `self.x_min` and `self.x_max`, divided into 
        `self.out_features + 1` intervals. The left bounds are set to the start 
        of each interval, and the right bounds are set to the end of each interval.
        If `self.use_importance_params` is True, the weights `self.w` are 
        initialized using Kaiming uniform initialization.
        """

        domain = torch.linspace(self.x_min, self.x_max, self.out_features+1)
        self.left_bounds.data = domain[:-1].clone().unsqueeze(0).repeat(self.in_features, 1)
        self.right_bounds.data = domain[1:].clone().unsqueeze(0).repeat(self.in_features, 1)
        if self.use_importance_params:
            nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))


    def forward(self, x):
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after applying the layer's transformations.

        Raises:
            AssertionError: If the input tensor does not have 2 dimensions.
        """

        assert len(x.shape) == 2, "Please check dimensions!"

        upper_bound = torch.max(self.right_bounds).item()
        lower_bound = torch.min(self.left_bounds).item()

        x = torch.tanh(x)
        x = lower_bound + 0.5*(x+1)*(upper_bound-lower_bound)

        is_lower_close = torch.allclose(x, torch.clamp(x, min=lower_bound), atol=self.eps)
        is_upper_close = torch.allclose(x, torch.clamp(x, max=upper_bound), atol=self.eps)

        assert is_lower_close and is_upper_close

        x = x.unsqueeze(2)

        left_bounds = self.left_bounds.unsqueeze(0)  
        right_bounds = self.right_bounds.unsqueeze(0)  

        active_hills = ((x >= left_bounds) & (x < right_bounds)).int()

        left_bounds = left_bounds * active_hills
        right_bounds = right_bounds * active_hills

        width = right_bounds - left_bounds
        norm_const = torch.where(width == 0, 0.0, 4 / (width**2 + self.eps))

        x = active_hills * torch.relu(x - left_bounds) \
            * torch.relu(right_bounds - x) \
            * norm_const
        x = x * x  

        if self.use_importance_params:
            x = x * self.w

        x = x.sum(dim=1)

        return x


    def extra_repr(self):
        """
        Returns a string representation of the layer's configuration.
        This method provides a detailed description of the layer's key attributes,
        which can be useful for debugging and logging purposes.
        
        Returns:
            str: A string containing the in_features, out_features, x_min, x_max,
                 train_domain, and use_importance_params attributes of the layer.
        """

        return (f"in_features={self.in_features}, "
            f"out_features={self.out_features}, " 
            f"x_min={self.x_min}, x_max={self.x_max}, "
            f"train_domain={self.left_bounds.requires_grad}, "
            f"use_importance_params={self.use_importance_params}")
    

    def get_slice(self, old_nclasses):
        return (slice(None), slice(None, old_nclasses))