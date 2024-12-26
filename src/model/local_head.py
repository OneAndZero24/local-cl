import torch
from torch import nn
import torch.nn.functional as F


class LocalHead(nn.Module):
    def __init__(self, 
        in_features: int,
        initial_out_features: int,
        x_min: float = -1.,
        x_max: float = 1.
    ):
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
    
    def increment(self, new_classes: list[int]):
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
        self.left_bounds.data = new_left_bounds.to(self.device)
        self.right_bounds.data = new_right_bounds.to(self.device)