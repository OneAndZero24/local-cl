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
        dtype = None
    ):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.in_features = in_features
        self.out_features = out_features
        self.train_domain = train_domain

        self.x_min = x_min
        self.x_max = x_max

        # Generate grid and corresponding parameters
        domain = torch.linspace(x_min, x_max, out_features+1)
        left_bounds  = domain[:-1].clone().unsqueeze(0).repeat(in_features,1)
        right_bounds = domain[1:].clone().unsqueeze(0).repeat(in_features,1)
        
        # Learnable parameters
        self.left_bounds = nn.Parameter(data=left_bounds, requires_grad=train_domain).to(self.device)
        self.right_bounds = nn.Parameter(data=right_bounds, requires_grad=train_domain).to(self.device)
        self.linear = nn.Linear(in_features, in_features).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):

        assert len(x.shape) == 2, "Please check dimensions!"

        # Flatten input
        x = x.flatten(start_dim=1)
        x = self.linear(x)

        upper_bound = float(torch.max(self.right_bounds))
        lower_bound = float(torch.min(self.left_bounds))

        # To project x onto domain of the hills
        x = F.hardtanh(x, min_val=lower_bound, max_val=upper_bound)

        # For stability
        eps = 1e-4 if self.train_domain else 0.0

        x = x.unsqueeze(2).repeat(1,1,self.out_features)
        norm_const = 4 / (self.right_bounds - self.left_bounds + eps)**2
        
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
    