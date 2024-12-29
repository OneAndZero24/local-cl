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
        self.train_domain = train_domain

        self.x_min = x_min
        self.x_max = x_max

        self.left_bounds = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs), requires_grad=train_domain)
        self.right_bounds = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs), requires_grad=train_domain)
        self.reset_parameters()

    def reset_parameters(self):
        domain = torch.linspace(self.x_min, self.x_max, self.out_features+1)
        self.left_bounds.data = domain[:-1].clone().unsqueeze(0).repeat(self.in_features, 1)
        self.right_bounds.data = domain[1:].clone().unsqueeze(0).repeat(self.in_features, 1)

    def forward(self, x, eps: float = 1e-8):

        assert len(x.shape) == 2, "Please check dimensions!"
        
        if not self.train_domain:
            eps = 0.0

        upper_bound = float(torch.max(self.right_bounds))
        lower_bound = float(torch.min(self.left_bounds))

        # To project x onto domain of the hills
        x = F.hardtanh(x, min_val=lower_bound, max_val=upper_bound)

        x = x.unsqueeze(2).repeat(1,1,self.out_features)
        norm_const = 4 / ((self.right_bounds - self.left_bounds)**2 + eps)
        
        # Forward pass
        x = torch.relu(x - self.left_bounds) \
            * torch.relu(self.right_bounds - x) \
            * norm_const
        
        x = x * x
        x = x.sum(dim=1)      

        return x

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
            f"out_features={self.out_features}, " 
            f"x_min={self.x_min}, x_max={self.x_max}, "
            f"train_domain={self.left_bounds.requires_grad}")
    