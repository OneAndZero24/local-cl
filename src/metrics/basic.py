from typing import Optional

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy, EpochAccuracy
from avalanche.evaluation.metrics.loss import EpochLoss
from avalanche.training.strategies.base_strategy import BaseStrategy

from .generic import MetricNamingMixin


class TotalLoss(MetricNamingMixin[float], EpochLoss):
    def __str__(self):
        return "Loss/total"


class TrainAccuracy(MetricNamingMixin[float], EpochAccuracy):
    def __str__(self):
        return "Accuracy"


class EvalAccuracy(MetricNamingMixin[float], GenericPluginMetric[float]):
    """Evaluation accuracy.

    Overrides Avalanche logic to be task-agnostic.
    Each experience will be reported separately with single task id of `0`.

    """

    def __init__(self):
        self._accuracy = Accuracy()
        self._metric: Accuracy
        GenericPluginMetric.__init__(self, self._accuracy, reset_at="experience", emit_at="experience", mode="eval")  # type: ignore

    def reset(self, strategy: Optional[BaseStrategy] = None) -> None:  # type: ignore
        self._metric.reset(0)  # type: ignore

    def result(self, strategy: Optional[BaseStrategy] = None) -> dict[int, float]:  # type: ignore
        return self._metric.result(0)  # type: ignore

    def update(self, strategy: BaseStrategy) -> None:
        self._accuracy.update(strategy.mb_output, strategy.mb_y, 0)  # type: ignore

    def __str__(self):
        return "Accuracy"
