import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LocalLayer(nn.Module):
    """
    Fully connected layer with local activation property.
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
        use_w=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_w = use_w

        self.eps = eps if train_domain else 0.0

        self.in_features = in_features
        self.out_features = out_features
        self.train_domain = train_domain

        self.x_min = x_min
        self.x_max = x_max

        self.left_bounds = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs), requires_grad=train_domain)
        self.right_bounds = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs), requires_grad=train_domain)
        if use_w:
            self.w = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        domain = torch.linspace(self.x_min, self.x_max, self.out_features+1)
        self.left_bounds.data = domain[:-1].clone().unsqueeze(0).repeat(self.in_features, 1)
        self.right_bounds.data = domain[1:].clone().unsqueeze(0).repeat(self.in_features, 1)
        if self.use_w:
            nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):

        assert len(x.shape) == 2, "Please check dimensions!"

        upper_bound = float(torch.max(self.right_bounds))
        lower_bound = float(torch.min(self.left_bounds))
        
        x = F.hardtanh(x, min_val=lower_bound, max_val=upper_bound)

        x = x.unsqueeze(2).repeat(1,1,self.out_features)
        left_bounds = self.left_bounds.unsqueeze(0).repeat(x.shape[0], 1, 1)
        right_bounds = self.right_bounds.unsqueeze(0).repeat(x.shape[0], 1, 1)

        active_hills = (x >= left_bounds) & (x < right_bounds)
        active_hills = active_hills.int()
        
        x = active_hills * x
        left_bounds = self.left_bounds * active_hills
        right_bounds = self.right_bounds * active_hills
       
        width = right_bounds - left_bounds
        norm_const = torch.where(width == 0, 0.0, 4 / ((width)**2 + self.eps))
        # TODO Zrobić najpierw sumowanie, potem aktywacja!!!!
        # Forward pass
        x = torch.relu(x - left_bounds) \
            * torch.relu(right_bounds - x) \
            * norm_const
        
        x = x * x
         
        if self.use_w:
            x = x * self.w
        x = x.sum(dim=1)    
       
        return x

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
            f"out_features={self.out_features}, " 
            f"x_min={self.x_min}, x_max={self.x_max}, "
            f"train_domain={self.left_bounds.requires_grad}")
    