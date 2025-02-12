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

from typing import Callable

from .local_module import LocalModule


def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Radial basis functions
def rbf_gaussian(x):
    return (-x.pow(2)).exp()


def rbf_linear(x):
    return x


def rbf_multiquadric(x):
    return (1 + x.pow(2)).sqrt()


def rbf_inverse_quadratic(x):
    return 1 / (1 + x.pow(2))


def rbf_inverse_multiquadric(x):
    return 1 / (1 + x.pow(2)).sqrt()


def rbf_bump(x):
    if torch.all(x.abs() <= 1):
        return (-1/(1-x.pow(2))).exp()
    return torch.zeros_like(x)


class RBFLayer(LocalModule):
    """
    A Radial Basis Function (RBF) Layer.

    An RBF layer is defined by the following elements:
        1. A radial kernel function `phi`.
        2. A positive shape parameter `epsilon`.
        3. The number of kernels `N` and their centers `c_i`, where `i=1, ..., N`.
        4. A norm function `||.||`.
        5. A set of weights `w_i`, where `i=1, ..., N`.

    The output of an RBF layer is given by:
        y(x) = sum_{i=1}^N a_i * phi(eps_i * ||x - c_i||)

    For more information, refer to:
        [1] https://en.wikipedia.org/wiki/Radial_basis_function
        [2] https://en.wikipedia.org/wiki/Radial_basis_function_network

    Args:
        in_features (int): Dimensionality of the input features.
        num_kernels (int): Number of kernels to use.
        out_features (int): Dimensionality of the output features.
        radial_function (Callable[[torch.Tensor], torch.Tensor]): A radial basis function that returns a tensor of real values given a tensor of real values.
        norm_function (Callable[[torch.Tensor], torch.Tensor]): Normalization function applied to the features.
        normalization (bool, optional): If True, applies the normalization trick to the RBF layer. Default is True.
        local_linear (bool, optional): If True, applies a trainable linear transformation to x-c. Default is False.
        initial_shape_parameter (torch.Tensor, optional): Sets the shape parameter to the desired value. Default is None.
        initial_centers_parameter (torch.Tensor, optional): Sets the centers to the desired value. Default is None.
        initial_weights_parameters (torch.Tensor, optional): Sets the weights parameter to the desired value. Default is None.
        constant_shape_parameter (bool, optional): Sets the shape parameters to a non-learnable constant. `initial_shape_parameter` must be provided if True. Default is False.
        constant_centers_parameter (bool, optional): Sets the centers to a non-learnable constant. `initial_centers_parameter` must be provided if True. Default is False.
        constant_weights_parameters (bool, optional): Sets the weights to a non-learnable constant. `initial_weights_parameters` must be provided if True. Default is False.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_kernels: int,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 norm_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True,
                 local_linear: bool = False,
                 initial_shape_parameter: torch.Tensor = None,
                 initial_centers_parameter: torch.Tensor = None,
                 initial_weights_parameters: torch.Tensor = None,
                 constant_shape_parameter: bool = False,
                 constant_centers_parameter: bool = False,
                 constant_weights_parameters: bool = False):
        super(RBFLayer, self).__init__()

        self.in_features = in_features
        self.num_kernels = num_kernels
        self.out_features = out_features
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization
        self.local_linear = local_linear

        self.initial_shape_parameter = initial_shape_parameter
        self.constant_shape_parameter = constant_shape_parameter

        self.initial_centers_parameter = initial_centers_parameter
        self.constant_centers_parameter = constant_centers_parameter

        self.initial_weights_parameters = initial_weights_parameters
        self.constant_weights_parameters = constant_weights_parameters

        assert radial_function is not None  \
            and norm_function is not None
        assert normalization is False or normalization is True

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialize linear combination weights
        if self.constant_weights_parameters:
            self.weights = nn.Parameter(
                self.initial_weights_parameters, requires_grad=False)
        else:
            self.weights = nn.Parameter(
                torch.zeros(
                    self.out_features,
                    self.num_kernels,
                    dtype=torch.float32))

        # Initialize kernels' centers
        if self.constant_centers_parameter:
            self.kernels_centers = nn.Parameter(
                self.initial_centers_parameter, requires_grad=False)
        else:
            self.kernels_centers = nn.Parameter(
                torch.zeros(
                    self.num_kernels,
                    self.in_features,
                    dtype=torch.float32))

        # Initialize shape parameter
        if self.constant_shape_parameter:
            self.log_shapes = nn.Parameter(
                self.initial_shape_parameter, requires_grad=False)
        else:
            self.log_shapes = nn.Parameter(
                torch.zeros(self.num_kernels, dtype=torch.float32))

        if self.local_linear:
            self.local_linear_weights = nn.Parameter(
                torch.zeros(self.num_kernels, self.in_features, dtype=torch.float32)
            )
            self.local_linear_bias = nn.Parameter(
                torch.zeros(self.num_kernels, dtype=torch.float32)
            )

        self.reset()

    def reset(self,
              lower_bound_kernels: float = 0.0,
              lower_bound_shapes: float = 0.5,
              upper_bound_kernels: float = 1.0,
              upper_bound_shapes: float = 1.0,
              gain_weights: float = 1.0) -> None:
        """
        Resets the parameters of the RBF layer.

        Args:
            upper_bound_kernels (float, optional): Upper bound for the uniform distribution used to initialize the kernel centers. Default is 1.0.
            std_shapes (float, optional): Standard deviation for the normal distribution used to initialize the log-shape parameters. Default is 0.1.
            gain_weights (float, optional): Gain for the Xavier uniform initialization of the weights. Default is 1.0.
        """
        if self.initial_centers_parameter is None:
            nn.init.uniform_(self.kernels_centers, a=lower_bound_kernels, b=upper_bound_kernels)

        if self.initial_shape_parameter is None:
            nn.init.uniform_(self.log_shapes, a=lower_bound_shapes, b=upper_bound_shapes)

        if self.initial_weights_parameters is None:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)

        if self.local_linear:
            nn.init.xavier_uniform_(self.local_linear_weights, gain=gain_weights)
            nn.init.zeros_(self.local_linear_bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RBF layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """

        # Input has size B x Fin
        batch_size = input.size(0)

        # Compute difference from centers
        # c has size B x num_kernels x Fin
        c = self.kernels_centers.expand(batch_size, self.num_kernels,
                                        self.in_features)

        diff = input.view(batch_size, 1, self.in_features) - c

        # Apply local linear transformation
        if self.local_linear:
            local_outputs = torch.matmul(diff, self.local_linear_weights)
            local_outputs = local_outputs + self.local_linear_bias

        # Apply norm function; c has size B x num_kernels
        r = self.norm_function(diff)

        # Apply parameter, eps_r has size B x num_kernels
        eps_r = self.log_shapes.exp().expand(batch_size, self.num_kernels) * r

        # Apply radial basis function; rbf has size B x num_kernels
        rbfs = self.radial_function(eps_r)

        if self.local_linear:
            rbfs = rbfs * local_outputs

        # Apply normalization
        # (check https://en.wikipedia.org/wiki/Radial_basis_function_network)
        if self.normalization:
            # 1e-9 prevents division by 0
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1)).unsqueeze(-1)

        # Take linear combination
        out = self.weights.expand(batch_size, self.out_features,
                                  self.num_kernels) * rbfs.view(
                                      batch_size, 1, self.num_kernels)

        return out.sum(dim=-1)

    @property
    def get_kernels_centers(self):
        """ Returns the centers of the kernels """
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        """ Returns the linear combination weights """
        return self.weights.detach()

    @property
    def get_shapes(self):
        """ Returns the shape parameters """
        return self.log_shapes.detach().exp()
    
    def incrementable_params(self):
        """ Returns the incrementable parameters of the module. """
        return ["weights"]
    