import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalConv2DLayer(nn.Module):
    """
    2D convolution layer with local activation property.
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
        domain = torch.linspace(self.x_min, self.x_max, self.out_channels+1)
        self.left_bounds.data = domain[:-1].clone().reshape((-1,1,1,1)).repeat(1, self.in_channels, self.size, self.size)
        self.right_bounds.data = domain[1:].clone().reshape((-1,1,1,1)).repeat(1, self.in_channels, self.size, self.size)

    def forward(self, x):
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
        return f"in_features={self.in_features}, 
            out_features={self.out_features},
            size={self.size},
            stride={self.stride}, 
            x_min={self.x_min}, x_max={self.x_max}, 
            train_domain={self.left_bounds.requires_grad}"
    