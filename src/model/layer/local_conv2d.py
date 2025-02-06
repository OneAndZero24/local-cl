import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_module import LocalModule
from util import deprecation_warning


class LocalConv2DLayer(LocalModule):
    """
    DEPRECATED  

    A 2D convolutional layer with local receptive fields and domain-specific bounds.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        train_domain (bool, optional): If True, the domain bounds are trainable. Default is True.
        x_min (float, optional): Minimum value of the domain. Default is -1.0.
        x_max (float, optional): Maximum value of the domain. Default is 1.0.
        device (torch.device, optional): The device on which to allocate the tensors. Default is None.
        dtype (torch.dtype, optional): The desired data type of the tensors. Default is None.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        x_min (float): Minimum value of the domain.
        x_max (float): Maximum value of the domain.
        left_bounds (torch.nn.Parameter): Left bounds of the domain for each output channel.
        right_bounds (torch.nn.Parameter): Right bounds of the domain for each output channel.

    Methods:
        reset_parameters():
            Initializes the domain bounds.
        forward(x):
            Applies the local convolution to the input tensor `x`.
        extra_repr():
            Returns a string with extra representation of the layer.
    """


    def __init__(self, 
        in_channels: int, 
        out_channels: int,
        size: int,
        stride: int = 1,
        train_domain: bool = True,
        x_min: float = -1.0,
        x_max: float = 1.0,
        device = None, 
        dtype = None
    ):
        """
        Initializes the LocalConv2DLayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            size (int): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Default is 1.
            train_domain (bool, optional): Whether the domain parameters should be trainable. Default is True.
            x_min (float, optional): Minimum value for the input range. Default is -1.0.
            x_max (float, optional): Maximum value for the input range. Default is 1.0.
            device (optional): The device on which to allocate the tensors. Default is None.
            dtype (optional): The desired data type of the tensors. Default is None.
        """
                
        deprecation_warning("LocalConv2DLayer is deprecated!")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.stride = stride
        self.x_min = x_min
        self.x_max = x_max

        self.left_bounds = nn.Parameter(torch.empty((out_channels, in_channels, size, size), **factory_kwargs), requires_grad=train_domain)
        self.right_bounds = nn.Parameter(torch.empty((out_channels, in_channels, size, size), **factory_kwargs), requires_grad=train_domain)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the local convolutional layer.
        This method initializes the left and right bounds of the convolutional
        layer based on a linear space between x_min and x_max. The bounds are
        reshaped and repeated to match the dimensions required for the layer's
        parameters.
        The left bounds are set to the values from the start of the domain to
        the second-to-last value, while the right bounds are set to the values
        from the second value to the end of the domain.
        The domain is divided into (out_channels + 1) parts to ensure that each
        output channel has a corresponding interval.

        Attributes:
            left_bounds (torch.Tensor): The left bounds of the intervals for each
                output channel, repeated to match the input channels and size.
            right_bounds (torch.Tensor): The right bounds of the intervals for each
                output channel, repeated to match the input channels and size.
        """

        domain = torch.linspace(self.x_min, self.x_max, self.out_channels+1)
        self.left_bounds.data = domain[:-1].clone().reshape((-1,1,1,1)).repeat(1, self.in_channels, self.size, self.size)
        self.right_bounds.data = domain[1:].clone().reshape((-1,1,1,1)).repeat(1, self.in_channels, self.size, self.size)

    def forward(self, x):
        """
        Perform the forward pass of the local convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the local convolution, 
                          with shape (batch_size, out_channels, new_height, new_width).
        The function performs the following steps:
        1. Extracts patches from the input tensor.
        2. Reshapes and repeats the patches to match the output channels.
        3. Applies a series of operations including ReLU and hardtanh activations.
        4. Computes the final result by summing over specific dimensions and reshaping.

        Note:
            - `self.size` and `self.stride` are used to determine the size and stride of the patches.
            - `self.out_channels` specifies the number of output channels.
            - `self.left_bounds`, `self.right_bounds`, `self.x_min`, and `self.x_max` are used in the activation functions.
        """

        patches = x.unfold(-2, self.size, self.stride).unfold(-1, self.size, self.stride)
        patches = patches.contiguous().view(*x.shape[:2], -1, self.size, self.size)
        patches = patches.unsqueeze(5).repeat(1,1,1,1,1,self.out_channels)
        patches = patches.permute(0, 2, 5, 1, 3, 4) # batch_size, nb_windows, out_channels, in_channels, size, size
       
        norm_const = 4 / (self.right_bounds - self.left_bounds)**2

        patches = patches.unsqueeze(2)
        res = (torch.relu(F.hardtanh(patches - self.left_bounds, min_val=self.x_min, max_val=self.x_max)) \
            * torch.relu(F.hardtanh(self.right_bounds - patches, min_val=self.x_min, max_val=self.x_max)) \
            * norm_const)**2

        res = res.sum([4, 5, 6]).permute(0, 3, 1, 2)
        h = int(res.size(2)**0.5)
        return res.view(res.shape[0], self.out_channels, h, -1)

    def extra_repr(self):
        """
        Returns a string representation of the layer's configuration.
        This method provides a detailed description of the layer's parameters,
        which can be useful for debugging and logging purposes.
        
        Returns:
            str: A string containing the in_features, out_features, size, stride,
                 x_min, x_max, and train_domain attributes of the layer.
        """

        return (f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"size={self.size}, "
            f"stride={self.stride}, " 
            f"x_min={self.x_min}, x_max={self.x_max}, " 
            f"train_domain={self.left_bounds.requires_grad}")
    