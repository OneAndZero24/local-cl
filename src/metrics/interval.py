from dataclasses import fields
from types import GenericAlias
from typing import Any, Callable, Optional

import numpy as np
import torch
import wandb
import wandb.viz
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
from avalanche.evaluation.metrics.loss import LossPluginMetric
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch import Tensor

from src.models.interval import IntervalMLP, IntervalModel
from ..strategies.interval import IntervalTraining

from .generic import MetricNamingMixin


class RobustAccuracy(MetricNamingMixin[float], AccuracyPluginMetric):
    """Evaluation accuracy based on robust model outputs.

    Reset on each experience (epoch).

    """

    def __init__(self):
        super().__init__(reset_at="experience", emit_at="experience", mode="eval")  # type: ignore

    def __str__(self):
        return "RobustAccuracy"

    def update(self, strategy: IntervalTraining):
        task_labels: list[Any] = strategy.experience.task_labels  # type: ignore
        if len(task_labels) > 1:
            task_label: Any = strategy.mb_task_id  # type: ignore
        else:
            task_label = task_labels[0]

        with torch.no_grad():
            robust_output = strategy.robust_output()

        self._accuracy.update(robust_output, strategy.mb_y, task_label)  # type: ignore


class Reporter(MetricNamingMixin[Tensor], LossPluginMetric):
    """Metric wrapper around IntervalTraining attributes."""

    def __init__(
        self,
        metric_name: str,
        strategy_attribute: str,
        strategy_subattribute: Optional[str] = None,
        strategy_attribute_key: Optional[str] = None,
        reset_at: str = "epoch",
        emit_at: str = "epoch",
        mode: str = "train",
    ):
        self.metric_name = metric_name
        self.strategy_attribute = strategy_attribute
        self.strategy_subattribute = strategy_subattribute
        self.strategy_attribute_key = strategy_attribute_key

        super().__init__(reset_at=reset_at, emit_at=emit_at, mode=mode)  # type: ignore

    def update(self, strategy: BaseStrategy) -> None:
        task_labels: list[Any] = strategy.experience.task_labels  # type: ignore
        if len(task_labels) > 1:
            task_label = 0
        else:
            task_label = task_labels[0]

        attr = getattr(strategy, self.strategy_attribute)
        attr = attr if self.strategy_subattribute is None else getattr(attr, self.strategy_subattribute)
        attr = attr if self.strategy_attribute_key is None else attr[self.strategy_attribute_key]
        self._loss.update(attr, patterns=len(strategy.mb_y), task_label=task_label)  # type: ignore

    def __str__(self):
        return f"{self.metric_name}"


class LayerDiagnostics(MetricNamingMixin[Tensor], GenericPluginMetric[Tensor]):
    """Wandb histogram metrics of a given layer's parameter tensor."""

    def __init__(
        self,
        layer_name: str,
        start: float = 0,
        stop: float = 1,
        n_bins: int = 100,
        reset_at: str = "epoch",
        emit_at: str = "epoch",
        mode: str = "train",
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        grad: bool = False,
    ):
        self.layer_name = layer_name
        self.start = start
        self.stop = stop
        self.n_bins = n_bins
        self.transform = transform
        self.grad = grad

        assert self.n_bins <= 512, "W&B does not support that many bins for visualization."

        self.data: Optional[Tensor] = None
        self.data_grad: Optional[Tensor] = None
        super().__init__(self.data, reset_at=reset_at, emit_at=emit_at, mode=mode)  # type: ignore

    def update(self, strategy: BaseStrategy) -> None:
        values: Optional[Tensor] = None
        for name, param in strategy.model.named_parameters():
            if name == self.layer_name:
                values = param
                break

        if values is not None:
            self.data = values.detach().cpu()
            if values.grad is not None:
                self.data_grad = values.grad.detach().cpu()

    def get_histogram(self) -> Optional[tuple[np.ndarray, np.ndarray]]:  # type: ignore
        if self.data is None or (self.grad and self.data_grad is None):
            return None

        bins = np.linspace(self.start, self.stop, num=self.n_bins)  # type: ignore

        if self.grad:
            assert self.data_grad is not None
            data = self.data_grad
        else:
            data = self.data
            if self.transform is not None:
                data = self.transform(data)

        data = data.view(-1).numpy()
        return np.histogram(data, bins=bins)  # type: ignore

    def result(self, strategy: BaseStrategy) -> Optional[wandb.Histogram]:  # type: ignore
        histogram = self.get_histogram()  # type: ignore
        return wandb.Histogram(np_histogram=histogram) if histogram is not None else None

    def reset(self, strategy: BaseStrategy) -> None:  # type: ignore
        self.data = None

    def __str__(self):
        return f"Diagnostics/{self.layer_name}" + (".grad" if self.grad else "")


class LayerDiagnosticsHist(LayerDiagnostics):
    """Raw histogram visualizations of a given layer's parameter tensor."""

    def __init__(
        self,
        layer_name: str,
        start: float = 0,
        stop: float = 1.0,
        n_bins: int = 20,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__(layer_name, start=start, stop=stop, n_bins=n_bins, transform=transform)

    def __str__(self):
        return f"DiagnosticsHist/{self.layer_name}"

    def _get_metric_name(self, strategy: BaseStrategy, add_experience: bool = True, add_task: Any = True):
        return super()._get_metric_name(strategy, add_experience=True, add_task=add_task)

    def result(self, strategy: BaseStrategy) -> Optional[wandb.viz.CustomChart]:  # type: ignore
        hist = self.get_histogram()  # type: ignore
        if hist is None:
            return None

        data: list[list[Any]] = []
        for i in range(len(hist[0])):  # type: ignore
            data.append([hist[0][i], f"{i:02d}: [{hist[1][i]:+.2f}, {hist[1][i+1]:+.2f}]"])

        table = wandb.Table(data=data, columns=["count", "bin"])
        title = self._get_metric_name(strategy, add_experience=True, add_task=False)
        return wandb.plot.bar(table, "bin", "count", title=title)  # type: ignore


def interval_training_diagnostics(model: IntervalModel):
    """Combined metrics for interval training."""

    metrics: list[Any] = []
    metrics.append(RobustAccuracy())

    losses = IntervalTraining.Losses()
    status = IntervalTraining.Status()

    metrics.extend([Reporter(f"Loss/{field.name}", "losses", field.name) for field in fields(losses)])
    if isinstance(model, IntervalMLP) and False:  # Do not run all diagnostics for CNN archs
    # if isinstance(model, IntervalModel):
        metrics.extend(
            [
                Reporter(f"Status/{field.name}", "status", field.name)
                for field in fields(status)
                if issubclass(field.type, Tensor)
            ]
        )
        metrics.extend(
            [
                Reporter(f"Status/{field.name}{layer}", "status", field.name, layer)
                for layer, _ in model.named_interval_children()
                for field in fields(status)
                if field.name == 'radius_mean_'
            ]
        )

        metrics.extend(
            [
                Reporter(f"Status/{field.name}{act_name}", "status", field.name, act_name)
                for act_name in model.output_names
                for field in fields(status)
                if field.name == 'bounds_width_'
            ]
        )

        metrics.extend(
            [
                LayerDiagnostics(layer, transform=model.radius_transform, stop=model.max_radius)
                for layer in model.state_dict().keys()
                if "radius" in layer
            ]
        )
        metrics.extend(
            [
                LayerDiagnosticsHist(layer, transform=model.radius_transform, stop=model.max_radius)
                for layer in model.state_dict().keys()
                if "radius" in layer
            ]
        )
        metrics.extend([LayerDiagnostics(layer, grad=True) for layer in model.state_dict().keys() if "radius" in layer])

    return metrics
