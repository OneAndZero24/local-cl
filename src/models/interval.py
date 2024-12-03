import math
from abc import ABC
from enum import Enum
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.dynamic import MultiTaskModule
from rich import print
from torch import Tensor
from torch.nn.parameter import Parameter


RADIUS_MIN = 0.

class Mode(Enum):
    VANILLA = 0
    EXPANSION = 1
    CONTRACTION_SHIFT = 2
    CONTRACTION_SCALE = 3


class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()


class PointLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.mode = Mode.VANILLA

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        with torch.no_grad():
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

        w_middle_pos = self.weight.clamp(min=0)
        w_middle_neg = self.weight.clamp(max=0)

        lower = x_lower @ w_middle_pos.t() + x_upper @ w_middle_neg.t() + self.bias
        upper = x_upper @ w_middle_pos.t() + x_lower @ w_middle_neg.t() + self.bias
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t() + self.bias

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.weight, self.bias])

        if mode == Mode.VANILLA:
            enable([self.weight, self.bias])
        elif mode == Mode.EXPANSION:
            pass
        elif mode == Mode.CONTRACTION_SHIFT:
            enable([self.weight, self.bias])
        elif mode == Mode.CONTRACTION_SCALE:
            pass


class IntervalLinear(IntervalModuleWithWeights):
    def __init__(
            self, in_features: int, out_features: int,
            radius_multiplier: float, max_radius: float, bias: bool,
            normalize_shift: bool, normalize_scale: bool, scale_init: float = -5.
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.radius_multiplier = radius_multiplier
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.scale_init = scale_init

        assert self.radius_multiplier > 0
        assert self.max_radius > 0

        self.weight = Parameter(torch.empty((out_features, in_features)))
        self._radius = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self._shift = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self._scale = Parameter(torch.empty((out_features, in_features)), requires_grad=False)

        # TODO test and fix so that it still works with bias=False
        if bias:
            self.bias = Parameter(torch.empty(out_features), requires_grad=True)
            self._bias_radius = Parameter(torch.empty_like(self.bias), requires_grad=False)
            self._bias_shift = Parameter(torch.empty_like(self.bias), requires_grad=False)
            self._bias_scale = Parameter(torch.empty_like(self.bias), requires_grad=False)
        else:
            self.bias = None
        self.mode: Mode = Mode.VANILLA
        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        return (params * torch.tensor(self.radius_multiplier)).clamp(min=RADIUS_MIN, max=self.max_radius + 0.1)

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @property
    def bias_radius(self) -> Tensor:
        return self.radius_transform(self._bias_radius)

    @property
    def shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._shift.device)
            return (self._shift / torch.max(self.radius, eps)).tanh()
        else:
            return self._shift.tanh()

    @property
    def bias_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._bias_shift.device)
            return (self._bias_shift / torch.max(self.bias_radius, eps)).tanh()
        else:
            return self._bias_shift.tanh()

    @property
    def scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._scale.device)
            scale = (self._scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._scale.sigmoid()
        return scale * (1.0 - torch.abs(self.shift))

    @property
    def bias_scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._bias_scale.device)
            scale = (self._bias_scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._bias_scale.sigmoid()
        return scale * (1.0 - torch.abs(self.bias_shift))

    def clamp_radii(self) -> None:
        with torch.no_grad():
            max = self.max_radius / self.radius_multiplier
            self._radius.clamp_(min=RADIUS_MIN, max=max)
            self._bias_radius.clamp_(min=RADIUS_MIN, max=max)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
            self._radius.fill_(self.max_radius)
            self._shift.zero_()
            self._scale.fill_(self.scale_init)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

                self.bias.zero_()
                self._bias_radius.fill_(self.max_radius)
                self._bias_shift.zero_()
                self._bias_scale.fill_(self.scale_init)

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.weight, self.bias, self._radius, self._shift, self._scale, self._bias_radius, self._bias_shift,
                 self._bias_scale])

        if mode == Mode.VANILLA:
            enable([self.weight, self.bias])
        elif mode == Mode.EXPANSION:
            with torch.no_grad():
                self._radius.fill_(self.max_radius)
                if self.bias is not None:
                    self._bias_radius.fill_(self.max_radius)
            enable([self._radius, self._bias_radius])
        elif mode == Mode.CONTRACTION_SHIFT:
            enable([self._shift, self._bias_shift])
        elif mode == Mode.CONTRACTION_SCALE:
            enable([self._scale, self._bias_scale])

    def freeze_task(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.weight + self.shift * self.radius)
            self._radius.copy_(self.scale * self._radius)
            self._shift.zero_()
            self._scale.fill_(self.scale_init)
            if self.bias is not None:
                self.bias.copy_(self.bias + self.bias_shift * self.bias_radius)
                self._bias_radius.copy_(self.bias_scale * self._bias_radius)
                self._bias_shift.zero_()
                self._bias_scale.fill_(self.scale_init)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

        if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
            w_middle: Tensor = self.weight
            w_lower = self.weight - self.radius
            w_upper = self.weight + self.radius
        else:
            assert self.mode in [Mode.CONTRACTION_SHIFT, Mode.CONTRACTION_SCALE]
            assert (0.0 <= self.scale).all() and (self.scale <= 1.0).all(), "Scale must be in [0, 1] range."
            assert (-1.0 <= self.shift).all() and (self.shift <= 1.0).all(), "Shift must be in [-1, 1] range."

            w_middle = self.weight + self.shift * self.radius
            w_lower = w_middle - self.scale * self.radius
            w_upper = w_middle + self.scale * self.radius

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)
        # Further splits only needed for numeric stability with asserts
        w_middle_pos = w_middle.clamp(min=0)
        w_middle_neg = w_middle.clamp(max=0)

        lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
        upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

        if self.bias is not None:
            b_middle = self.bias + self.bias_shift * self.bias_radius
            b_lower = b_middle - self.bias_scale * self.bias_radius
            b_upper = b_middle + self.bias_scale * self.bias_radius
            lower = lower + b_lower
            upper = upper + b_upper
            middle = middle + b_middle

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore

class DeIntervaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        return x_middle

class ReIntervaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.rename(None)
        tiler = [1] * (len(x.shape) + 1)
        tiler[1] = 3
        x = x.unsqueeze(1).tile(tiler)
        return x
        

class IntervalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1. / (1 - self.p)

    def forward(self, x):
        if self.training:
            x = x.refine_names("N", "bounds", ...)
            x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)),
                                             x.unbind("bounds"))  # type: ignore
            mask = torch.bernoulli(self.p * torch.ones_like(x_middle)).long()
            x_lower = x_lower.where(mask != 1, torch.zeros_like(x_lower)) * self.scale
            x_middle = x_middle.where(mask != 1, torch.zeros_like(x_middle)) * self.scale
            x_upper = x_upper.where(mask != 1, torch.zeros_like(x_upper)) * self.scale
            return torch.stack([x_lower, x_middle, x_upper], dim=1)
        else:
            return x


class IntervalMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalBatchNorm2d(IntervalModuleWithWeights):
    def __init__(self, num_features,
                 interval_statistics: bool = False,
                 affine: bool = True,
                 normalize_shift: bool = True,
                 momentum: float = 0.1,
                 scale_init: float = 5.0,
                 max_radius: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.interval_statistics = interval_statistics
        self.affine = affine
        self.scale_init = scale_init
        self.max_radius = max_radius
        self.momentum = momentum

        self.normalize_shift = normalize_shift
        self.normalize_scale = False

        self.epsilon = 1e-5
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self._gamma_radius = Parameter(torch.empty(num_features), requires_grad=False)
            self._gamma_shift = Parameter(torch.empty(num_features), requires_grad=False)
            self._gamma_scale = Parameter(torch.empty(num_features), requires_grad=False)

            self.beta = nn.Parameter(torch.zeros(num_features))
            self._beta_radius = Parameter(torch.empty(num_features), requires_grad=False)
            self._beta_shift = Parameter(torch.empty(num_features), requires_grad=False)
            self._beta_scale = Parameter(torch.empty(num_features), requires_grad=False)

        self.mode: Mode = Mode.VANILLA
        self.register_buffer('running_mean', torch.zeros(num_features, requires_grad=False))
        self.register_buffer('running_var', torch.ones(num_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.affine:
            return

        with torch.no_grad():
            self.beta.zero_()
            self._beta_radius.fill_(self.max_radius)
            self._beta_shift.zero_()
            self._beta_scale.fill_(self.scale_init)

            self.gamma.fill_(1.)
            self._gamma_radius.fill_(self.max_radius)
            self._gamma_shift.zero_()
            self._gamma_scale.fill_(self.scale_init)

    def freeze_task(self) -> None:
        if not self.affine:
            return

        with torch.no_grad():
            self.gamma.copy_(self.gamma + self.gamma_shift * self.gamma_radius)
            self.beta.copy_(self.beta + self.beta_shift * self.beta_radius)

            self._gamma_radius.copy_(self.gamma_scale * self._gamma_radius)
            self._beta_radius.copy_(self.beta_scale * self._beta_radius)

            self._beta_shift.zero_()
            self._gamma_shift.zero_()

            self._gamma_scale.fill_(5)
            self._beta_scale.fill_(5)

    def clamp_radii(self) -> None:
        if not self.affine:
            return
        with torch.no_grad():
            max = self.max_radius / self.radius_multiplier
            self._gamma_radius.clamp_(min=RADIUS_MIN, max=max)
            self._beta_radius.clamp_(min=RADIUS_MIN, max=max)

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.gamma, self._gamma_radius, self._gamma_shift, self._gamma_scale,
                 self.beta, self._beta_radius, self._beta_shift, self._beta_scale])

        if mode == Mode.VANILLA:
            enable([self.gamma, self.beta])
        elif mode == Mode.CONTRACTION_SHIFT:
            enable([self._gamma_shift, self._beta_shift])
        elif mode == Mode.CONTRACTION_SCALE:
            enable([self._gamma_scale, self._beta_scale])

    def radius_transform(self, params: Tensor):
        return (params * torch.tensor(self.radius_multiplier)).clamp(min=RADIUS_MIN, max=self.max_radius + 0.1)

    @property
    def gamma_radius(self) -> Tensor:
        return self.radius_transform(self._gamma_radius)

    @property
    def beta_radius(self) -> Tensor:
        return self.radius_transform(self._beta_radius)

    @property
    def gamma_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._gamma_shift.device)
            return (self._gamma_shift / torch.max(self._gamma_radius, eps)).tanh()
        else:
            return self._gamma_shift.tanh()

    @property
    def beta_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._beta_shift.device)
            return (self._beta_shift / torch.max(self._beta_radius, eps)).tanh()
        else:
            return self._beta_shift.tanh()

    @property
    def gamma_scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._gamma_scale.device)
            scale = (self._gamma_scale / torch.max(self._gamma_radius, eps)).sigmoid()
        else:
            scale = self._gamma_scale.sigmoid()
        return scale * (1.0 - torch.abs(self.gamma_shift))

    @property
    def beta_scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._beta_scale.device)
            scale = (self._beta_scale / torch.max(self._beta_radius, eps)).sigmoid()
        else:
            scale = self._beta_scale.sigmoid()
        return scale * (1.0 - torch.abs(self.beta_shift))

    def forward(self, x):
        x = x.refine_names("N", "bounds", "C", "H", "W")  # type: ignore
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

        if self.interval_statistics and self.training:
            # Calculating whitening nominator: x - E[x]
            mean_lower = x_lower.mean([0, 2, 3], keepdim=True)
            mean_middle = x_middle.mean([0, 2, 3], keepdim=True)
            mean_upper = x_upper.mean([0, 2, 3], keepdim=True)

            nominator_upper = x_upper - mean_lower
            nominator_middle = x_middle - mean_middle
            nominator_lower = x_lower - mean_upper

            # Calculating denominator: sqrt(Var[x] + eps)
            # Var(x) = E[x^2] - E[x]^2
            mean_squared_lower = torch.where(
                torch.logical_and(x_lower <= 0, 0 <= x_upper),
                torch.zeros_like(x_middle),
                torch.minimum(x_upper ** 2, x_lower ** 2)).mean([0, 2, 3], keepdim=True)
            mean_squared_middle = (x_middle ** 2).mean([0, 2, 3], keepdim=True)
            mean_squared_upper = torch.maximum(x_upper ** 2, x_lower ** 2).mean([0, 2, 3], keepdim=True)

            squared_mean_lower = torch.where(
                torch.logical_and(mean_lower <= 0, 0 <= mean_upper),
                torch.zeros_like(mean_middle),
                torch.minimum(mean_lower ** 2, mean_upper ** 2))
            squared_mean_middle = mean_middle ** 2
            squared_mean_upper = torch.maximum(mean_lower ** 2, mean_upper ** 2)

            var_lower = mean_squared_lower - squared_mean_upper
            var_middle = mean_squared_middle - squared_mean_middle
            var_upper = mean_squared_upper - squared_mean_lower

            assert torch.all(var_lower <= var_middle)
            assert torch.all(var_middle <= var_upper)

            # TODO: Just clip?
            var_lower = torch.clamp(var_lower, min=0)
            assert torch.all(var_lower >= 0.), "Variance has to be non-negative"
            assert torch.all(var_middle >= 0.), "Variance has to be non-negative"
            assert torch.all(var_upper >= 0.), "Variance has to be non-negative"

            denominator_lower = (var_lower + self.epsilon).sqrt()
            denominator_middle = (var_middle + self.epsilon).sqrt()
            denominator_upper = (var_upper + self.epsilon).sqrt()

            # Dividing nominator by denominator
            whitened_lower = torch.where(nominator_lower > 0,
                                         nominator_lower / denominator_upper,
                                         nominator_lower / denominator_lower)
            whitened_middle = nominator_middle / denominator_middle
            whitened_upper = torch.where(nominator_upper > 0,
                                         nominator_upper / denominator_lower,
                                         nominator_upper / denominator_upper)

        else:

            if self.training:
                mean_middle = x_middle.mean([0, 2, 3], keepdim=True)
                var_middle = x_middle.var([0, 2, 3], keepdim=True)
            else:
                mean_middle = self.running_mean.view(1, -1, 1, 1)
                var_middle = self.running_var.view(1, -1, 1, 1)

            nominator_lower = x_lower - mean_middle
            nominator_middle = x_middle - mean_middle
            nominator_upper = x_upper - mean_middle

            denominator = (var_middle + self.epsilon).sqrt()

            whitened_lower = nominator_lower / denominator
            whitened_middle = nominator_middle / denominator
            whitened_upper = nominator_upper / denominator

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_middle.view(-1).detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_middle.view(-1).detach()

        assert (whitened_lower <= whitened_middle).all()
        assert (whitened_middle <= whitened_upper).all()

        if self.affine:

            if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
                gamma_middle = self.gamma
                gamma_lower = self.gamma - self.gamma_radius
                gamma_upper = self.gamma + self.gamma_radius

                beta_middle = self.beta
                beta_lower = self.beta - self.gamma_radius
                beta_upper = self.beta + self.gamma_radius
            else:
                assert self.mode in [Mode.CONTRACTION_SHIFT, Mode.CONTRACTION_SCALE]
                assert (0.0 <= self.gamma_scale).all() and (
                        self.beta_scale <= 1.0).all(), "Scale must be in [0, 1] range."
                assert (-1.0 <= self.gamma_shift).all() and (
                        self.beta_shift <= 1.0).all(), "Shift must be in [-1, 1] range."
                gamma_middle = self.gamma + self.gamma_shift * self.gamma_radius
                gamma_lower = gamma_middle - self.gamma_scale * self.gamma_radius
                gamma_upper = gamma_middle + self.gamma_scale * self.gamma_radius

                beta_middle = self.beta + self.beta_shift * self.beta_radius
                beta_lower = beta_middle - self.beta_scale * self.beta_radius
                beta_upper = beta_middle + self.beta_scale * self.beta_radius

            gamma_lower = gamma_lower.view(1, -1, 1, 1)
            gamma_middle = gamma_middle.view(1, -1, 1, 1)
            gamma_upper = gamma_upper.view(1, -1, 1, 1)

            gammafied_all = torch.stack([
                gamma_lower * whitened_lower,
                gamma_lower * whitened_upper,
                gamma_upper * whitened_lower,
                gamma_upper * whitened_upper], dim=-1)
            gammafied_lower, _ = gammafied_all.min(-1)
            gammafied_middle = gamma_middle * whitened_middle
            gammafied_upper, _ = gammafied_all.max(-1)

            beta_lower = beta_lower.view(1, -1, 1, 1)
            beta_middle = beta_middle.view(1, -1, 1, 1)
            beta_upper = beta_upper.view(1, -1, 1, 1)

            final_lower = gammafied_lower + beta_lower
            final_middle = gammafied_middle + beta_middle
            final_upper = gammafied_upper + beta_upper

            assert (final_lower <= final_middle).all()
            assert (final_middle <= final_upper).all()

            return torch.stack([final_lower, final_middle, final_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                                             "W")
        else:
            return torch.stack([whitened_lower, whitened_middle, whitened_upper], dim=1).refine_names("N", "bounds",
                                                                                                      "C", "H", "W")


class IntervalAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super().__init__(output_size)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalConv2d(nn.Conv2d, IntervalModuleWithWeights):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            radius_multiplier: float,
            max_radius: float,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            normalize_shift: bool = True,
            normalize_scale: bool = False,
            scale_init: float = -5.
    ) -> None:
        IntervalModuleWithWeights.__init__(self)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.radius_multiplier = radius_multiplier
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.scale_init = scale_init

        assert self.radius_multiplier > 0
        assert self.max_radius > 0

        self._radius = Parameter(torch.empty_like(self.weight), requires_grad=False)
        self._shift = Parameter(torch.empty_like(self.weight), requires_grad=False)
        self._scale = Parameter(torch.empty_like(self.weight), requires_grad=False)
        # TODO test and fix so that it still works with bias=False
        if bias:
            self._bias_radius = Parameter(torch.empty_like(self.bias), requires_grad=False)
            self._bias_shift = Parameter(torch.empty_like(self.bias), requires_grad=False)
            self._bias_scale = Parameter(torch.empty_like(self.bias), requires_grad=False)
        self.mode: Mode = Mode.VANILLA
        self.init_parameters()

    # TODO abstract away this logic in a common base class?
    def radius_transform(self, params: Tensor):
        return (params * torch.tensor(self.radius_multiplier)).clamp(min=RADIUS_MIN, max=self.max_radius)

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @property
    def bias_radius(self) -> Tensor:
        return self.radius_transform(self._bias_radius)

    @property
    def shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._shift.device)
            return (self._shift / torch.max(self.radius, eps)).tanh()
        else:
            return self._shift.tanh()

    @property
    def bias_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._bias_shift.device)
            return (self._bias_shift / torch.max(self.bias_radius, eps)).tanh()
        else:
            return self._bias_shift.tanh()

    @property
    def scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._scale.device)
            scale = (self._scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._scale.sigmoid()
        return scale * (1.0 - torch.abs(self.shift))

    @property
    def bias_scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._bias_scale.device)
            scale = (self._bias_scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._bias_scale.sigmoid()
        return scale * (1.0 - torch.abs(self.bias_shift))

    def clamp_radii(self) -> None:
        with torch.no_grad():
            max = self.max_radius / self.radius_multiplier
            self._radius.clamp_(min=RADIUS_MIN, max=max)
            if self.bias is not None:
                self._bias_radius.clamp_(min=RADIUS_MIN, max=max)

    def init_parameters(self) -> None:
        with torch.no_grad():
            # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
            self._radius.fill_(RADIUS_MIN)
            self._shift.zero_()
            self._scale.fill_(5)
            if self.bias is not None:
                # self.bias.zero_()
                self._bias_radius.fill_(RADIUS_MIN)
                self._bias_shift.zero_()
                self._bias_scale.fill_(5)

    def switch_mode(self, mode: Mode) -> None:
        if self.mode == Mode.VANILLA and mode == Mode.CONTRACTION_SCALE:
            self._radius.fill_(self.max_radius)
            if self.bias is not None:
                self._bias_radius.fill_(self.max_radius)
        self.mode = mode

        def enable(params: list[Parameter]):
            for p in params:
                p.requires_grad_()

        def disable(params: list[Parameter]):
            for p in params:
                p.requires_grad_(False)
                p.grad = None

        disable([self.weight, self._radius, self._shift, self._scale])
        if self.bias is not None:
            disable([self.bias, self._bias_radius, self._bias_shift, self._bias_scale])


        if mode == Mode.VANILLA:
            enable([self.weight, self.bias])
        elif mode == Mode.CONTRACTION_SHIFT:
            enable([self._shift])
            if self.bias is not None:
                enable([self._bias_shift])
        elif mode == Mode.CONTRACTION_SCALE:
            enable([self._scale])
            if self.bias is not None:
                enable([self._bias_scale])

    def freeze_task(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.weight + self.shift * self.radius)
            self._radius.copy_(self.scale * self._radius)
            self._shift.zero_()
            self._scale.fill_(5)
            if self.bias is not None:
                self.bias.copy_(self.bias + self.bias_shift * self.bias_radius)
                self._bias_radius.copy_(self.bias_scale * self._bias_radius)
                self._bias_shift.zero_()
                self._bias_scale.fill_(5)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "C", "H", "W")
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

        if self.mode in [Mode.VANILLA, Mode.EXPANSION]:
            w_middle: Tensor = self.weight
            w_lower = self.weight - self.radius
            w_upper = self.weight + self.radius
            if self.bias is not None:
                b_middle = self.bias
                b_lower = b_middle - self.bias_radius
                b_upper = b_middle + self.bias_radius
        else:
            assert self.mode in [Mode.CONTRACTION_SHIFT, Mode.CONTRACTION_SCALE]
            assert (0.0 <= self.scale).all() and (self.scale <= 1.0).all(), "Scale must be in [0, 1] range."
            assert (-1.0 <= self.shift).all() and (self.shift <= 1.0).all(), "Shift must be in [-1, 1] range."
            w_middle = self.weight + self.shift * self.radius
            w_lower = w_middle - self.scale * self.radius
            w_upper = w_middle + self.scale * self.radius
            if self.bias is not None:
                b_middle = self.bias + self.bias_shift * self.bias_radius
                b_lower = b_middle - self.bias_scale * self.bias_radius
                b_upper = b_middle + self.bias_scale * self.bias_radius

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)
        # Further splits only needed for numeric stability with asserts
        w_middle_neg = w_middle.clamp(max=0)
        w_middle_pos = w_middle.clamp(min=0)

        l_lp = F.conv2d(x_lower, w_lower_pos, None, self.stride, self.padding, self.dilation, self.groups)
        u_ln = F.conv2d(x_upper, w_lower_neg, None, self.stride, self.padding, self.dilation, self.groups)
        u_up = F.conv2d(x_upper, w_upper_pos, None, self.stride, self.padding, self.dilation, self.groups)
        l_un = F.conv2d(x_lower, w_upper_neg, None, self.stride, self.padding, self.dilation, self.groups)
        m_mp = F.conv2d(x_middle, w_middle_pos, None, self.stride, self.padding, self.dilation, self.groups)
        m_mn = F.conv2d(x_middle, w_middle_neg, None, self.stride, self.padding, self.dilation, self.groups)

        lower = l_lp + u_ln
        upper = u_up + l_un
        middle = m_mp + m_mn
        # numerically unstable?
        # middle = F.conv2d(x_middle, w_middle, None, self.stride, self.padding, self.dilation, self.groups)

        if self.bias is not None:
            lower = lower + b_lower.view(1, b_lower.size(0), 1, 1)
            upper = upper + b_upper.view(1, b_upper.size(0), 1, 1)
            middle = middle + b_middle.view(1, b_middle.size(0), 1, 1)

        # assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        # assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        # Safety net for rare numerical errors.
        if not (lower <= middle).all():
            diff = torch.where(lower > middle, lower - middle, torch.zeros_like(middle)).abs().sum()
            print(f"Lower bound must be less than or equal to middle bound. Diff: {diff}")
            lower = torch.where(lower > middle, middle, lower)
        if not (middle <= upper).all():
            diff = torch.where(middle > upper, middle - upper, torch.zeros_like(middle)).abs().sum()
            print(f"Middle bound must be less than or equal to upper bound. Diff: {diff}")
            upper = torch.where(middle > upper, middle, upper)

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore


class IntervalModel(MultiTaskModule):
    def __init__(self, radius_multiplier: float, max_radius: float):
        super().__init__()

        self.mode: Mode = Mode.VANILLA
        self._radius_multiplier = radius_multiplier
        self._max_radius = max_radius

    def interval_children(self) -> list[IntervalModuleWithWeights]:
        return [m for m in self.modules() if isinstance(m, IntervalModuleWithWeights)]

    def named_interval_children(self) -> list[tuple[str, IntervalModuleWithWeights]]:
        # TODO: hack
        return [("last" if "last" in n else n, m)
                for n, m in self.named_modules()
                if isinstance(m, IntervalModuleWithWeights) and not isinstance(m, IntervalBatchNorm2d)]

    def switch_mode(self, mode: Mode) -> None:
        if mode == Mode.VANILLA:
            print("\n[bold cyan]» :green_circle: Switching to vanilla training phase...")
        elif mode == Mode.EXPANSION:
            print("\n[bold cyan]» :yellow circle: Switching to interval expansion phase...")
        elif mode == Mode.CONTRACTION_SHIFT:
            print("\n[bold cyan]» :heavy_large_circle: Switching to shift contraction phase...")
        elif mode == Mode.CONTRACTION_SCALE:
            print("\n[bold cyan]» :heavy_large_circle: Switching to scale contraction phase...")

        self.mode = mode
        for m in self.interval_children():
            m.switch_mode(mode)

        for m in self.last:
            if isinstance(m, PointLinear):
                m.switch_mode(mode)

    def freeze_task(self) -> None:
        for m in self.interval_children():
            m.freeze_task()

    @property
    def radius_multiplier(self):
        return self._radius_multiplier

    @radius_multiplier.setter
    def radius_multiplier(self, value: float):
        self._radius_multiplier = value
        for m in self.interval_children():
            m.radius_multiplier = value

    @property
    def max_radius(self):
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = value
        for m in self.interval_children():
            m.max_radius = value

    def clamp_radii(self) -> None:
        for m in self.interval_children():
            m.clamp_radii()

    def radius_transform(self, params: Tensor) -> Tensor:
        for m in self.interval_children():
            return m.radius_transform(params)

        raise ValueError("No IntervalNet modules found in model.")


class IntervalMLP(IntervalModel):
    def __init__(
            self,
            input_size: int,
            hidden_dim: int,
            output_classes: int,
            radius_multiplier: float,
            max_radius: float,
            bias: bool,
            heads: int,
            normalize_shift: bool,
            normalize_scale: bool,
            scale_init: float,
    ):
        super().__init__(radius_multiplier=radius_multiplier, max_radius=max_radius)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.output_names = ['fc1', 'fc2']
        self.fc1 = IntervalLinear(
            self.input_size, self.hidden_dim,
            radius_multiplier=radius_multiplier, max_radius=max_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
            scale_init=scale_init
        )
        self.fc2 = IntervalLinear(
            self.hidden_dim, self.hidden_dim,
            radius_multiplier=radius_multiplier, max_radius=max_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
            scale_init=scale_init,
        )
        if heads > 1:
            # Incremental task, we don't have to use intervals
            self.last = nn.ModuleList([
                PointLinear(self.hidden_dim, self.output_classes) for _ in range(heads)
            ])
        else:
            self.last = nn.ModuleList([
                IntervalLinear(
                    self.hidden_dim,
                    self.output_classes,
                    radius_multiplier=radius_multiplier,
                    max_radius=max_radius,
                    bias=bias,
                    normalize_shift=normalize_shift,
                    normalize_scale=normalize_scale,
                    scale_init=scale_init,
                )])

    # MW: this is a modified function from avalanche
    def forward(self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample.
        :return:
        """
        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)

        full_out = {}
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if not full_out:
                for key, val in out_task.items():
                    full_out[key] = torch.empty(x.shape[0], *val.shape[1:],
                                                device=val.device).rename(None)
            for key, val in out_task.items():
                full_out[key][task_mask] = val.rename(None)

        for key, val in full_out.items():
            full_out[key] = val.refine_names("N", "bounds", "features")
        return full_out

    def forward_base(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        x = x.refine_names("N", "C", "H", "W")  # type: ignore  # expected input shape
        x = x.rename(None)  # type: ignore  # drop names for unsupported operations
        x = x.flatten(1)  # (N, features)
        x = x.unflatten(1, (1, -1))  # type: ignore  # (N, bounds, features)
        x = x.tile((1, 3, 1))

        x = x.refine_names("N", "bounds", "features")  # type: ignore

        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))

        return {
            "fc1": fc1,
            "fc2": fc2,
        }

    def forward_single_task(self, x: Tensor, task_id: int) -> dict[str, Tensor]:
        # Get activations from the second-to-last layer
        activation_dict = self.forward_base(x)
        activation_dict["last"] = self.last[task_id](activation_dict["fc2"])
        return activation_dict

    @property
    def device(self):
        return self.fc1.weight.device


class IntervalSimpleCNN(IntervalModel):
    def __init__(
            self,
            input_size: int,
            hidden_dim: int,
            output_classes: int,
            radius_multiplier: float,
            max_radius: float,
            bias: bool,
            heads: int,
            normalize_shift: bool,
            normalize_scale: bool,
            scale_init: float,
    ):
        super().__init__(radius_multiplier=radius_multiplier, max_radius=max_radius)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.output_names = ['features']

        cnn_layers = []

        in_channels = 3
        out_channels = 64
        for layer_idx in range(5):
            cnn_layers.append(
                IntervalConv2d(in_channels, out_channels, kernel_size=3, stride=2 if layer_idx % 2 else 1, padding=1, groups=1,
                               radius_multiplier=self.radius_multiplier, max_radius=self.max_radius, bias=True,
                               normalize_shift=self.normalize_shift, normalize_scale=self.normalize_scale,
                               scale_init=scale_init)
            )
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.hidden_dim = 4096

        if heads > 1:
            # Incremental task, we don't have to use intervals
            self.last = nn.ModuleList([
                PointLinear(self.hidden_dim, self.output_classes) for _ in range(heads)
            ])
        else:
            self.last = nn.ModuleList([
                IntervalLinear(
                    self.hidden_dim,
                    self.output_classes,
                    radius_multiplier=radius_multiplier,
                    max_radius=max_radius,
                    bias=bias,
                    normalize_shift=normalize_shift,
                    normalize_scale=normalize_scale,
                    scale_init=scale_init,
                )])

    # MW: this is a modified function from avalanche
    def forward(self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample.
        :return:
        """
        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)

        full_out = {}
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if not full_out:
                for key, val in out_task.items():
                    full_out[key] = torch.empty(x.shape[0], *val.shape[1:],
                                                device=val.device).rename(None)
            for key, val in out_task.items():
                full_out[key][task_mask] = val.rename(None)

        for key, val in full_out.items():
            full_out[key] = val.refine_names("N", "bounds", "features")
        return full_out

    def forward_base(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        x = self.cnn_layers(x)
        x = x.rename(None).flatten(2).refine_names("N", "bounds", "features")

        return {
            "features": x,
        }

    def forward_single_task(self, x: Tensor, task_id: int) -> dict[str, Tensor]:
        # Get activations from the second-to-last layer
        x = x.unsqueeze(1).tile(1, 3, 1, 1, 1)
        activation_dict = self.forward_base(x)
        activation_dict["last"] = self.last[task_id](activation_dict["features"])
        return activation_dict

    @property
    def device(self):
        return self.fc1.weight.device


BN_INTERVAL_STATISTICS = False


class ReLU6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(0, 6)
    
class IntervalAlexNet(IntervalModel):
    def __init__(self, in_channels, output_classes, heads,
                 radius_multiplier, max_radius,
                 normalize_shift, normalize_scale, scale_init, act_fn):
        super(IntervalAlexNet, self).__init__(
                radius_multiplier=radius_multiplier, max_radius=max_radius)

        self.output_classes = output_classes
        self.radius_multipler = radius_multiplier
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.scale_init = scale_init
        if act_fn == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif act_fn == "relu6":
            self.act_fn = ReLU6()
        else:
            raise NotImplementedError


        self.features = nn.Sequential(
            IntervalConv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                           radius_multiplier=self.radius_multiplier, max_radius=self.max_radius,
                           normalize_shift=self.normalize_shift, normalize_scale=self.normalize_scale,
                           scale_init=self.scale_init),
            self.act_fn,
            IntervalMaxPool2d(kernel_size=2),
            IntervalConv2d(64, 192, kernel_size=3, padding=1,
                           radius_multiplier=self.radius_multiplier, max_radius=self.max_radius,
                           normalize_shift=self.normalize_shift, normalize_scale=self.normalize_scale,
                           scale_init=self.scale_init),
            self.act_fn,
            IntervalMaxPool2d(kernel_size=2),
            IntervalConv2d(192, 384, kernel_size=3, padding=1,
                           radius_multiplier=self.radius_multiplier, max_radius=self.max_radius,
                           normalize_shift=self.normalize_shift, normalize_scale=self.normalize_scale,
                           scale_init=self.scale_init),
            self.act_fn,
            IntervalConv2d(384, 256, kernel_size=3, padding=1,
                           radius_multiplier=self.radius_multiplier, max_radius=self.max_radius,
                           normalize_shift=self.normalize_shift, normalize_scale=self.normalize_scale,
                           scale_init=self.scale_init),
            self.act_fn,
            IntervalConv2d(256, 256, kernel_size=3, padding=1,
                           radius_multiplier=self.radius_multiplier, max_radius=self.max_radius,
                           normalize_shift=self.normalize_shift, normalize_scale=self.normalize_scale,
                           scale_init=self.scale_init),
            self.act_fn,
            IntervalMaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            IntervalDropout(),
            IntervalLinear(256 * 2 * 2, 4096, bias=True,
                      radius_multiplier=radius_multiplier, max_radius=max_radius,
                      normalize_shift=normalize_shift, normalize_scale=normalize_scale,
                      scale_init=scale_init),
            self.act_fn,
            IntervalDropout(),
            IntervalLinear(4096, 4096, bias=True,
                      radius_multiplier=radius_multiplier, max_radius=max_radius,
                      normalize_shift=normalize_shift, normalize_scale=normalize_scale,
                      scale_init=scale_init),
            self.act_fn,
        )
        if heads > 1:
            # Incremental task, we don't have to use intervals
            self.last = nn.ModuleList([
                PointLinear(4096, self.output_classes) for _ in range(heads)
            ])
        else:
            self.last = nn.ModuleList([
                IntervalLinear(
                    4096,
                    output_classes,
                    radius_multiplier=self.radius_multiplier,
                    max_radius=self.max_radius,
                    bias=True,
                    normalize_shift=self.normalize_shift,
                    normalize_scale=self.normalize_scale,
                    scale_init=self.scale_init,
                )])

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = x.unsqueeze(1).tile(1, 3, 1, 1, 1)
        x = x.refine_names("N", "bounds", "C", "H", "W")  # type: ignore  # expected input shape
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 2 * 2)
        x = x.rename(None).flatten(2).refine_names("N", "bounds", "features")
        x = self.classifier(x)
        activation_dict = {}
        # activation_dict = {"features": x}
        x = self.last[task_id](x)
        activation_dict["last"] = x
        return activation_dict

    def forward(self, x: Tensor, task_labels: torch.Tensor) -> Tensor:
        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)
            if len(unique_tasks) == 1:
                return self.forward_single_task(x, task_labels[0])

        full_out = {}
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if not full_out:
                for key, val in out_task.items():
                    full_out[key] = torch.empty(x.shape[0], *val.shape[1:],
                                                device=val.device).rename(None)
            for key, val in out_task.items():
                full_out[key][task_mask] = val.rename(None)

        for key, val in full_out.items():
            full_out[key] = val.refine_names("N", "bounds", ...)
        return full_out
