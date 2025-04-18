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
import numpy as np

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
        2. A shape parameter `sigma_i`, where `i=1, ..., N`.
        3. The number of kernels `N` and their centers `c_i`, where `i=1, ..., N`.
        4. A norm function `||.||`.
        5. A set of weights `w_i`, where `i=1, ..., N`.

    The output of an RBF layer is given by:
        y(x) = sum_{i=1}^N a_i * phi(||(x - c_i)/sigma_i||)

    For more information, refer to:
        [1] https://en.wikipedia.org/wiki/Radial_basis_function
        [2] https://en.wikipedia.org/wiki/Radial_basis_function_network

    Args:
        in_features (int): Dimensionality of the input features.
        num_kernels (int): Number of kernels to use.
        out_features (int): Dimensionality of the output features.
        no_groups (int): Number of the neuron groups.
        no_mask_update_iterations (int): Number of iterations required for convergence to a full matrix of ones in the mask. This is used only when growing_mask is set to True.
        growing_mask (bool): If True, the number of ones in the mask is growing.
        radial_function (Callable[[torch.Tensor], torch.Tensor]): A radial basis function that returns a tensor of real values given a tensor of real values.
        norm_function (Callable[[torch.Tensor], torch.Tensor]): Normalization function applied to the features.
        normalization (bool, optional): If True, applies the normalization trick to the RBF layer. Default is True.
        local_linear (bool, optional): If True, applies a trainable linear transformation to x-c. Default is False.
        times_square (bool, optional): If True, multiplies x^2 by output. Default is False.
        initial_shape_parameter (torch.Tensor, optional): Sets the shape parameter to the desired value. Default is None.
        initial_centers_parameter (torch.Tensor, optional): Sets the centers to the desired value. Default is None.
        initial_weights_parameters (torch.Tensor, optional): Sets the weights parameter to the desired value. Default is None.
        constant_shape_parameter (bool, optional): Sets the shape parameters to a non-learnable constant. `initial_shape_parameter` must be provided if True. Default is False.
        constant_centers_parameter (bool, optional): Sets the centers to a non-learnable constant. `initial_centers_parameter` must be provided if True. Default is False.
        constant_weights_parameters (bool, optional): Sets the weights to a non-learnable constant. `initial_weights_parameters` must be provided if True. Default is False.
        start_empty (bool, optional): If True, the mask is initialized to zero and gradually increased to full ones. Default
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_kernels: int,
                 no_groups: int,
                 no_mask_update_iterations: int,
                 growing_mask: bool,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 norm_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True,
                 local_linear: bool = False,
                 times_square: bool = False,
                 initial_shape_parameter: torch.Tensor = None,
                 initial_centers_parameter: torch.Tensor = None,
                 initial_weights_parameters: torch.Tensor = None,
                 constant_shape_parameter: bool = False,
                 constant_centers_parameter: bool = False,
                 constant_weights_parameters: bool = False,
                 start_empty: bool = False):
        super(RBFLayer, self).__init__()

        self.in_features = in_features
        self.num_kernels = num_kernels
        self.no_groups = no_groups
        self.out_features = out_features
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization
        self.local_linear = local_linear
        self.times_square = times_square

        self.initial_shape_parameter = initial_shape_parameter
        self.constant_shape_parameter = constant_shape_parameter

        self.initial_centers_parameter = initial_centers_parameter
        self.constant_centers_parameter = constant_centers_parameter

        self.initial_weights_parameters = initial_weights_parameters
        self.constant_weights_parameters = constant_weights_parameters
        self.start_empty = start_empty

        assert radial_function is not None  \
            and norm_function is not None
        assert normalization is False or normalization is True

        self.no_mask_update_iterations = no_mask_update_iterations
        self.growing_mask = growing_mask

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
                torch.zeros(
                    self.num_kernels, 
                    self.in_features,
                    dtype=torch.float32))

        if self.local_linear:
            self.local_linear_weights = nn.Parameter(
                torch.zeros(self.out_features, self.num_kernels, self.in_features, dtype=torch.float32)
            )
            self.local_linear_bias = nn.Parameter(
                torch.zeros(self.out_features, self.num_kernels, dtype=torch.float32)
            )

        if self.times_square:
            self.square_scale = nn.Parameter(
                torch.ones(self.num_kernels, self.in_features, dtype=torch.float32)
            )

        # Initialize mask to define groups
        self.mask = self.init_group_mask()

        self.reset()

    def init_group_mask(self):
        """Initialize masks to define group of neurons within a neural network"""

        device = self.kernels_centers.device

        self.iteration = 0
        assert self.no_groups > 0, "Number of created groups should be greater than 0."
        group_size_neurons = self.num_kernels // self.no_groups
        group_size_features = self.in_features // self.no_groups

        group_size_features = self.in_features // self.no_groups 
        group_size_neurons = self.num_kernels // self.no_groups

        mask = torch.zeros((self.num_kernels, self.in_features))

        if not self.start_empty:
            for g in range(self.no_groups):
                start_feature = g * group_size_features
                end_feature = min((g + 1) * group_size_features, self.in_features)
                feature_indices = range(start_feature, end_feature)

                start_neuron = g * group_size_neurons
                end_neuron = min((g + 1) * group_size_neurons, self.num_kernels)
                neuron_indices = range(start_neuron, end_neuron)

                mask[np.ix_(list(neuron_indices), list(feature_indices))] = 1
                
            # Check if all neurons are used
            # unused_neurons = np.where(mask.sum(axis=1) == 0)[0]
            # assert len(unused_neurons) == 0, "There are unused neurons!"

        return nn.Parameter(mask.to(device), requires_grad=False)
    
    def update_mask(self):
        """Gradually increases the number of ones in a mask, guaranteeing full ones at the final epoch."""
        growth_factor = (self.iteration + 1) / self.no_mask_update_iterations
        target_ones = int(growth_factor * self.in_features)

        for neuron in range(self.num_kernels):
            current_active = torch.sum(self.mask[neuron]).item()
            additional_needed = max(0, target_ones - int(current_active))

            if additional_needed > 0:
                inactive_features = torch.where(self.mask[neuron] == 0)[0]
                if len(inactive_features) > 0:
                    chosen_features = inactive_features[torch.randperm(len(inactive_features))[:additional_needed]]
                    self.mask[neuron, chosen_features] = 1
        self.iteration += 1

        if self.iteration == self.no_mask_update_iterations - 1:
            self.mask[:, :] = 1

    def reset(self,
              lower_bound_kernels: float = 0.0,
              upper_bound_kernels: float = 1.0,
              log_shapes_std: float = 0.1,
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
            nn.init.normal_(self.log_shapes, mean=upper_bound_kernels/2, std=log_shapes_std)

        if self.initial_weights_parameters is None:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)

        if self.local_linear:
            nn.init.xavier_uniform_(self.local_linear_weights, gain=gain_weights)
            nn.init.constant_(self.local_linear_bias, 0.01)

        if self.times_square:
            nn.init.xavier_uniform_(self.square_scale, gain=gain_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RBF layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """

        if self.growing_mask and self.training and not (self.mask == torch.ones_like(self.mask)).all():
            self.update_mask()
            
        # Input has size B x Fin
        batch_size = input.size(0)

        # Compute difference from centers
        # c has size B x num_kernels x Fin
        c = self.kernels_centers.expand(batch_size, self.num_kernels,
                                        self.in_features)
        
        # Compute shape
        # sigma has size B x num_kernels x Fin
        sigma = self.log_shapes.exp().expand(batch_size, self.num_kernels,
                                             self.in_features)

        x = input.view(batch_size, 1, self.in_features)

        diff = (x - c) / sigma
        diff *= self.mask

        # Apply norm function; c has size B x num_kernels
        r = self.norm_function(diff)

        # Apply radial basis function; rbf has size B x num_kernels
        rbfs = self.radial_function(r)

        # Apply normalization
        # (check https://en.wikipedia.org/wiki/Radial_basis_function_network)
        if self.normalization:
            # 1e-9 prevents division by 0
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1)).unsqueeze(-1)

        weights = self.weights.expand(batch_size, self.out_features, self.num_kernels)

        # Apply local linear transformation
        if self.local_linear:
            weights = torch.einsum('bni,oni->bon', diff, self.local_linear_weights) + self.local_linear_bias

        # Apply linear combination; out has size B x Fout

        out = weights * rbfs.view(batch_size, 1, self.num_kernels)

        if self.times_square:
            square_scale = self.square_scale.expand(batch_size, self.num_kernels, self.in_features)
            sq = square_scale*(x*x)
            out = torch.einsum('bni,bon->bon', sq, out)

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
        
        R = ["weights"]
        if self.local_linear:
            R.append("local_linear_weights")
            R.append("local_linear_bias")
        return R