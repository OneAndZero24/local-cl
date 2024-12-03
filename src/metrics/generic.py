from typing import Any, TypeVar

from avalanche.evaluation.metric_definitions import GenericPluginMetric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import phase_and_task, stream_type
from avalanche.training.strategies.base_strategy import BaseStrategy

TResult = TypeVar("TResult")


def get_metric_name(
    metric: PluginMetric[Any],
    strategy: BaseStrategy,
    add_experience: bool = False,
    add_task: Any = True,
    count_experience_from_zero: bool = True,
):
    """Customized version of Avalanche metric name generator."""

    phase_name, task_label = phase_and_task(strategy)
    stream = stream_type(strategy.experience)  # type: ignore
    base_name = "{}/{}/{}".format(str(metric), phase_name, stream)
    experience: int = strategy.experience.current_experience  # type: ignore
    if not count_experience_from_zero:
        experience += 1
    exp_name = f"/exp{experience:01}"
    task_name = ""

    if task_label is None and isinstance(add_task, bool):
        add_task = False
    else:
        if isinstance(add_task, bool):
            if add_task:
                task_name = f"/Task{task_label:01}"
        elif isinstance(add_task, int):
            task_name = f"/Task{add_task:01}"
            add_task = True

    if add_experience and not add_task:
        return base_name + exp_name
    elif add_experience and add_task:
        return base_name + task_name + exp_name
    elif not add_experience and not add_task:
        return base_name
    elif not add_experience and add_task:
        return base_name + task_name


class MetricNamingMixin(GenericPluginMetric[TResult]):
    """Avalanche GenericPluginMetric with custom naming style."""

    def _get_metric_name(self, strategy: BaseStrategy, add_experience: bool = False, add_task: Any = True):
        return get_metric_name(
            self, strategy=strategy, add_experience=add_experience, add_task=add_task, count_experience_from_zero=False
        )

    def _package_result(self, strategy: BaseStrategy) -> MetricResult:
        metric_value: Any = self.result(strategy)  # type: ignore
        add_exp: bool = self._emit_at == "experience"  # type: ignore
        plot_x_position: int = self.get_global_counter()

        if isinstance(metric_value, dict):
            metrics: list[MetricValue] = []
            for k, v in metric_value.items():  # type: ignore
                if len(metric_value) == 1:  # type: ignore
                    k = False
                metric_name = self._get_metric_name(strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))  # type: ignore
            return metrics
        else:
            metric_name = self._get_metric_name(strategy, add_experience=add_exp, add_task=False)
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]  # type: ignore
