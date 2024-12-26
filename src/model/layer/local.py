import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalLayer(nn.Module):
    """
    Fully connected layer with local activation property.
    """

    def __init__(self, 
        in_features: int, 
        out_features: int,
        train_domain: bool = True,
        x_min: float = -1.0,
        x_max: float = 1.0,
        device = None, 
        dtype = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.x_min = x_min
        self.x_max = x_max
        
        self.left_bounds = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs), requires_grad=train_domain)
        self.right_bounds = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs), requires_grad=train_domain)
        self.reset_parameters()

    def reset_parameters(self):
        domain = torch.linspace(self.x_min, self.x_max, self.out_features+1)
        self.left_bounds.data = domain[:-1].clone().unsqueeze(0).repeat(self.in_features, 1)
        self.right_bounds.data = domain[1:].clone().unsqueeze(0).repeat(self.in_features, 1)

    def forward(self, x):
        x = x.unsqueeze(2).repeat(1, 1, self.out_features)
       
        norm_const = 4 / (self.right_bounds - self.left_bounds)**2

        x = (torch.relu(F.hardtanh(x - self.left_bounds, min_val=self.x_min, max_val=self.x_max)) \
            * torch.relu(F.hardtanh(self.right_bounds - x, min_val=self.x_min, max_val=self.x_max)) \
            * norm_const)**2

        return x.sum(dim=1)

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"x_min={self.x_min}, x_max={self.x_max}, "
            f"train_domain={self.left_bounds.requires_grad}")
    