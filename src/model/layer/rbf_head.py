# Based on: 
# PyTorchRBFLayer
# MIT License

# Copyright (c) 2021 Alessio Russo [alessior@kth.se]. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

from .local_module import LocalModule
from .rbf import l_norm, rbf_gaussian


class SingleRBFHeadLayer(LocalModule):
    """
    A Radial Basis Function (RBF) Layer as incremental head layer.

    An RBF layer is defined by the following elements:
        1. A radial kernel function `phi`. Here the kernel function is always set to be Gaussian.
        2. A shape parameter `sigma_i`, where `i=1, ..., N`.
        3. The number of kernels `N` and their centers `c_i`, where `i=1, ..., N`.
        4. A norm function `||.||`.

    The output of an RBF layer is given by:
        y(x) = (phi(||(x - c_1)/sigma_1||, ..., phi(||(x - c_N)/sigma_N||)   
    
    For more information, refer to:
        [1] https://en.wikipedia.org/wiki/Radial_basis_function
        [2] https://en.wikipedia.org/wiki/Radial_basis_function_network

    Args:
        in_features (int): Dimensionality of the input features.
        out_features (int): Dimensionality of the output features.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ):
        super(SingleRBFHeadLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.radial_function = rbf_gaussian
        self.norm_function = l_norm

        self._make_parameters()

    def _make_parameters(self) -> None:

        self.kernels_centers = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.float32))

        self.log_shapes = nn.Parameter(
            torch.zeros(
                self.out_features, 
                self.in_features,
                dtype=torch.float32))

        self.reset()


    def reset(self,
              lower_bound_kernels: float = 0.0,
              upper_bound_kernels: float = 1.0,
              log_shapes_std: float = 0.1) -> None:
        """
        Resets the parameters of the RBF layer.

        Args:
            upper_bound_kernels (float, optional): Upper bound for the uniform distribution used to initialize the kernel centers. Default is 1.0.
            std_shapes (float, optional): Standard deviation for the normal distribution used to initialize the log-shape parameters. Default is 0.1.
        """
        nn.init.uniform_(self.kernels_centers, a=lower_bound_kernels, b=upper_bound_kernels)
        nn.init.normal_(self.log_shapes, mean=upper_bound_kernels/2, std=log_shapes_std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RBF classification head layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Input has size B x Fin
        batch_size = input.size(0)

        # Compute difference from centers
        # c has size B x out_features x Fin
        c = self.kernels_centers.expand(batch_size, self.out_features,
                                        self.in_features)
        
        # Compute shape
        # sigma has size B x out_features x Fin
        sigma = self.log_shapes.exp().expand(batch_size, self.out_features,
                                             self.in_features)

        diff = (input.view(batch_size, 1, self.in_features) - c) / sigma

        # Apply norm function; c has size B x out_features
        r = self.norm_function(diff)

        # Apply radial basis function; rbf has size B x out_features
        rbfs = self.radial_function(r)
        return rbfs

    @property
    def get_kernels_centers(self):
        """ Returns the centers of the kernels """
        return self.kernels_centers.detach()

    @property
    def get_shapes(self):
        """ Returns the shape parameters """
        return self.log_shapes.detach().exp()
    
    def incrementable_params(self):
        """ Returns the incrementable parameters of the module. """
        return ["kernels_centers", "log_shapes"]