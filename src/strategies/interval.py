from collections import deque
from dataclasses import dataclass, InitVar, fields, field
from typing import Optional, Sequence, Any, cast

import numpy as np
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import visdom
import wandb
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from rich import print  # type: ignore # noqa
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics.functional.classification.accuracy import accuracy

from src.cfg import Settings
from src.models.interval import IntervalConv2d, IntervalMLP, Mode, IntervalBatchNorm2d
from src.strategies import VanillaTraining


class IntervalTraining(VanillaTraining):
    """Main interval training strategy."""

    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device: torch.device = torch.device("cpu"),
            plugins: Optional[Sequence[StrategyPlugin]] = None,
            evaluator: Optional[EvaluationPlugin] = None,
            eval_every: int = -1,
            enable_visdom: bool = False,
            visdom_reset_every_epoch: bool = False,
            *,
            cfg: Settings,
    ):
        super().__init__(  # type: ignore
            model,
            optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            cfg=cfg,
        )

        self.mb_output_all: dict[str, Tensor]
        """All model's outputs computed on the current mini-batch (lower, middle, upper bounds), per layer."""

        self.model: IntervalMLP

        self.scale_learning_rate = self.cfg.interval.scale_learning_rate

        # Training metrics for the current mini-batch
        self.losses: Optional[IntervalTraining.Losses] = None  # Reported as 'Loss/*' metrics
        self.status: Optional[IntervalTraining.Status] = None  # Reported as 'Status/*' metrics

        # Running metrics
        self._accuracy: deque[Tensor] = deque(
            maxlen=self.cfg.interval.metric_lookback
        )  # latest readings from the left
        self._robust_accuracy: deque[Tensor] = deque(
            maxlen=self.cfg.interval.metric_lookback
        )  # latest readings from the left

        self.model.radius_multiplier = self.cfg.interval.radius_multiplier
        self.model.max_radius = self.cfg.interval.max_radius

        enable_visdom = False
        self.viz = visdom.Visdom() if enable_visdom else None
        self.viz_debug = visdom.Visdom(env="debug") if enable_visdom else None
        self.viz_reset_every_epoch = visdom_reset_every_epoch
        self.windows: dict[str, str] = {}

    @property
    def mode(self):
        """Current phase of model training.

        Returns
        -------
        Mode
            VANILLA, EXPANSION or CONTRACTION.

        """
        return self.model.mode

    @property
    def mode_numeric(self) -> Tensor:
        """Current phase of model training converted to a float.

        Returns
        -------
        Tensor
            0 - VANILLA
            1 - EXPANSION
            2 - CONTRACTION

        """
        return torch.tensor(self.mode.value).float()

    def accuracy(self, n_last: int = 1) -> Tensor:
        """Moving average of the batch accuracy."""
        assert n_last <= self.cfg.interval.metric_lookback
        if not self._accuracy:
            return torch.tensor(0.0)
        return torch.stack(list(self._accuracy)[:n_last]).mean().detach().cpu()

    def robust_accuracy(self, n_last: int = 1) -> Tensor:
        """Moving average of the batch robust accuracy."""
        assert n_last <= self.cfg.interval.metric_lookback
        if not self._robust_accuracy:
            return torch.tensor(0.0)
        return torch.stack(list(self._robust_accuracy)[:n_last]).mean().detach().cpu()

    # ----------------------------------------------------------------------------------------------
    # Training hooks
    # ----------------------------------------------------------------------------------------------
    def after_forward(self, **kwargs: Any):
        """Rebind the model's default output to the middle bound."""
        assert isinstance(self.mb_output, dict)
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output["last"][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        """Rebind the model's default output to the middle bound."""
        assert isinstance(self.mb_output, dict)
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output["last"][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    @dataclass
    class Losses:
        """Model losses reported as 'Loss/*'."""

        total: Tensor = torch.tensor(0.0, device="cuda")
        vanilla: Tensor = torch.tensor(0.0, device="cuda")
        robust: Tensor = torch.tensor(0.0, device="cuda")

        radius_penalty: Tensor = torch.tensor(0.0, device="cuda")
        robust_penalty: Tensor = torch.tensor(0.0, device="cuda")
        bounds_penalty: Tensor = torch.tensor(0.0, device="cuda")

    # def criterion(self, ):
    #     if self.is_training:
    #         # Use class masking for incremental class training in the same way as Continual Learning Benchmark
    #         preds = self.mb_output[:, : self.valid_classes]
    #     else:
    #         preds = self.mb_output

    #     return self._criterion(preds, self.mb_y)

    def before_backward(self, **kwargs: Any):
        """Compute interval training losses."""
        super().before_backward(**kwargs)  # type: ignore

        self.losses = IntervalTraining.Losses()
        self.losses.vanilla = self.loss.clone().detach()
        robust_loss = self._criterion(self.robust_output(), self.mb_y)
        robust_penalty = robust_loss * 0.
        self.losses.robust = cast(Tensor, robust_loss.detach().clone())

        assert isinstance(self.mb_output, Tensor)
        self._accuracy.appendleft(accuracy(self.mb_output, self.mb_y))
        self._robust_accuracy.appendleft(accuracy(self.robust_output(), self.mb_y))

        if self.mode == Mode.VANILLA:
            total_loss = self.loss
        elif self.mode == Mode.CONTRACTION_SHIFT:
            total_loss = self.loss
        elif self.mode == Mode.CONTRACTION_SCALE:
            # ---------------------------------------------------------------------------------------------------------
            # Contraction phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust penalty ===
            if self.robust_accuracy(self.cfg.interval.metric_lookback) < (
                    self.cfg.interval.robust_accuracy_threshold * self.accuracy(self.cfg.interval.metric_lookback)
            ):
                robust_penalty = robust_loss
            else:
                robust_penalty = robust_loss * 0.0
            total_loss = robust_penalty

        # weights = torch.cat([m.weight.flatten() for m in self.model.interval_children()])
        # l1: Tensor = torch.linalg.vector_norm(weights, ord=1) / weights.shape[0]  # type: ignore
        # self.l1_penalty += self.l1_lambda * l1

        self.loss = total_loss  # Rebind as Avalanche loss
        self.losses.total = total_loss.detach().clone()  # Rebind as Avalanche loss
        self.losses.robust_penalty = robust_penalty.detach().clone()
        self.diagnostics()

    def after_backward(self, **kwargs: Any):
        super().after_backward(**kwargs)  # type: ignore

        pass  # Debugging breakpoint

    def after_update(self, **kwargs: Any):
        """Cleanup after each step."""
        super().after_update(**kwargs)  # type: ignore
        self.model.clamp_radii()


    def before_training_exp(self, **kwargs: Any):
        """Switch mode or freeze on each consecutive experience."""
        super().before_training_exp(**kwargs)  # type: ignore

        if self.training_exp_counter == 1:
            self.model.switch_mode(Mode.CONTRACTION_SHIFT)
            self.model.freeze_task()
            self.make_optimizer()
        elif self.training_exp_counter > 1:
            self.model.switch_mode(Mode.CONTRACTION_SHIFT)
            self.model.freeze_task()
            self.make_optimizer()

        self._accuracy.clear()
        self._robust_accuracy.clear()

    def _gather_radii_stats(self):
        prev_radius_sum: dict[str, float] = {}
        prev_radius_log_sum: dict[str, float] = {}

        current_radius_sum: dict[str, float] = {}
        current_radius_log_sum: dict[str, float] = {}


        for name, module in self.model.named_interval_children():
            if isinstance(module, IntervalBatchNorm2d):
                continue
            prev_radius_sum[name] = module.radius.sum().item()
            prev_radius_log_sum[name] = module.radius.log().sum().item()

            current_radius = module.radius * module.scale
            current_radius_sum[name] = current_radius.sum().item()
            current_radius_log_sum[name] = current_radius.log().sum().item()

            if module.bias is not None:
                prev_radius_sum[name + "_bias"] = module.bias_radius.sum().item()
                prev_radius_log_sum[name + "_bias"] = module.bias_radius.log().sum().item()

                current_bias_radius = module.bias_radius * module.bias_scale
                current_radius_sum[name + "_bias"] = current_bias_radius.sum().item()
                current_radius_log_sum[name + "_bias"] = current_bias_radius.log().sum().item()

        prev_radius_sum["all"] = sum(v for v in prev_radius_sum.values())
        prev_radius_log_sum["all"] = sum(v for v in prev_radius_log_sum.values())
        current_radius_sum["all"] = sum(v for v in current_radius_sum.values())
        current_radius_log_sum["all"] = sum(v for v in current_radius_log_sum.values())

        prev_radius_sum = {"prev_radius_sum/" + k: v for k, v in prev_radius_sum.items()}
        prev_radius_log_sum = {"prev_radius_log_sum/" + k: v for k, v in prev_radius_log_sum.items()}
        current_radius_sum = {"current_radius_sum/" + k: v for k, v in current_radius_sum.items()}
        current_radius_log_sum = {"current_radius_log_sum/" + k: v for k, v in current_radius_log_sum.items()}

        wandb.log(prev_radius_sum, commit=False)
        wandb.log(prev_radius_log_sum, commit=False)
        wandb.log(current_radius_sum, commit=False)
        wandb.log(current_radius_log_sum, commit=False)

    def before_training_epoch(self, **kwargs: Any):
        """Switch to expansion phase when ready."""
        super().before_training_epoch(**kwargs)  # type: ignore

        if self.mode in [Mode.VANILLA,
                         Mode.CONTRACTION_SHIFT] and self.epoch == self.train_epochs - self.cfg.interval.contraction_epochs:
            self.model.switch_mode(Mode.CONTRACTION_SCALE)
        if self.mode == Mode.CONTRACTION_SHIFT:
            if self.epoch == 0:
                self.optimizer.param_groups[0]["lr"] = self.cfg.interval.expansion_learning_rate  # type: ignore
            elif self.cfg.milestones is not None and self.epoch in self.cfg.milestones.keys():
                current_lr = self.optimizer.param_groups[0]["lr"]
                new_lr = current_lr * self.cfg.milestones[self.epoch]
                self.optimizer.param_groups[0]["lr"] = new_lr
        elif self.mode == Mode.CONTRACTION_SCALE:
            self.model.eval()
            self.optimizer.param_groups[0]["lr"] = self.cfg.interval.scale_learning_rate  # type: ignore

        if self.viz_debug:
            self.reset_viz_debug()
        self._gather_radii_stats()

    # ----------------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------------
    def make_train_dataloader(self, num_workers=4, shuffle=True,
                              pin_memory=True, **kwargs):
        """
        Called after the dataset adaptation. Initializes the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=True)

    def robust_output(self):
        """Get the robust version of the current output.

        Returns
        -------
        Tensor
            Robust output logits (lower bound for correct class, upper bounds for incorrect classes).

        """
        output_lower, _, output_higher = self.mb_output_all["last"].unbind("bounds")
        y_oh = F.one_hot(self.mb_y, num_classes=self.model.output_classes)  # type: ignore
        return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore

    def bounds_width(self, layer_name: str):
        """Compute the width of the activation bounds.

        Parameters
        ----------
        layer_name : str
            Name of the layer.

        Returns
        -------
        Tensor
            Difference between the upper and lower bounds of activations for a given layer.

        """
        bounds: Tensor = self.mb_output_all[layer_name].rename(None)  # type: ignore
        return bounds[:, 2, :] - bounds[:, 0, :]

    @dataclass
    class Status:
        """Diagnostic values reported as 'Status/*'."""

        mode: Tensor = torch.tensor(0.0)
        radius_multiplier: Tensor = torch.tensor(0.0)
        radius_mean: Tensor = torch.tensor(0.0)

        radius_mean_: dict[str, Tensor] = field(default_factory=lambda: {})
        radius_log_sum_: dict[str, Tensor] = field(default_factory=lambda: {})
        bounds_width_: dict[str, Tensor] = field(default_factory=lambda: {})

    def diagnostics(self):
        """Save training diagnostics before each update."""
        self.status = IntervalTraining.Status()
        self.status.mode = self.mode_numeric
        self.status.radius_multiplier = torch.tensor(self.cfg.interval.radius_multiplier)

        # radii: list[Tensor] = []

        # for name, module in self.model.named_interval_children():
        #     if isinstance(module, IntervalBatchNorm2d):
        #         continue
        #     radii.append((module.radius * module.scale).detach().cpu().flatten())
        #     self.status.radius_mean_[name] = radii[-1].mean()

        #     if self.viz and self.mb_it == len(self.dataloader) - 1:  # type: ignore
        #         self.windows[f"{name}.radius"] = self.viz.heatmap(
        #             module.radius,
        #             win=self.windows.get(f"{name}.radius"),
        #             opts={"title": f"{name}.radius --> epoch {(self.epoch or 0) + 1}"},
        #         )

        #         self.windows[f"{name}.weight"] = self.viz.heatmap(
        #             module.weight.abs().clamp(max=module.weight.abs().quantile(0.99)),
        #             win=self.windows.get(f"{name}.weight"),
        #             opts={"title": f"{name}.weight.abs() (w/o outliers) --> epoch {(self.epoch or 0) + 1}"},
        #         )

        # for name in self.mb_output_all.keys():
        #     self.status.bounds_width_[name] = self.bounds_width(name).mean()

        # self.status.radius_mean = torch.cat(radii).mean()

        if self.viz_debug:
            for (
                    metric,
                    name,
                    window,
                    _,
                    color,
                    dash,
                    yrange,
            ) in self.get_debug_metrics():
                self.append_viz_debug(metric, name, window, color, dash, yrange)

    def get_debug_metrics(self):
        """Return a list of batch debug metrics to visualize with Visdom plots.

        Returns
        -------
        list[tuple[ Tensor, str, str, str, tuple[int, int, int], str, tuple[float, float] ]]
            List of (metric, metric_name, window_name, window_title, linecolor) tuples.

        """

        epoch = f"(epoch: {(self.epoch or 0) + 1})"
        _ = torch.tensor(0.0)

        output_type = list[tuple[Tensor, str, str, str, tuple[int, int, int], str, tuple[float, float]]]

        metrics = cast(
            output_type,
            [
                (
                    self.robust_accuracy(1),
                    "robust_accuracy",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (7, 126, 143),
                    "solid",
                    (-0.1, 1.1),
                ),
                (
                    self.accuracy(1),
                    "accuracy",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (219, 0, 108),
                    "solid",
                    (-0.1, 1.1),
                ),
                (
                    self.robust_accuracy(self.cfg.interval.metric_lookback),
                    f"robust_accuracy_ma{self.cfg.interval.metric_lookback}",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (7, 126, 143),
                    "dot",
                    (-0.1, 1.1),
                ),
                (
                    self.accuracy(self.cfg.interval.metric_lookback),
                    f"accuracy_ma{self.cfg.interval.metric_lookback}",
                    "accuracy",
                    f"Batch accuracy {epoch}",
                    (219, 0, 108),
                    "dot",
                    (-0.1, 1.1),
                ),
                (
                    self.losses.robust_penalty if self.losses else _,
                    "robust_penalty",
                    "penalties",
                    f"Penalties {epoch}",
                    (7, 126, 143),
                    "solid",
                    (-0.1, self.cfg.interval.max_radius + 0.1),
                ),
                (
                    self.losses.radius_penalty if self.losses else _,
                    "radius_penalty",
                    "penalties",
                    f"Penalties {epoch}",
                    (230, 203, 0),
                    "solid",
                    (-0.1, self.cfg.interval.max_radius + 0.1),
                ),
                (
                    self.status.radius_mean if self.status else _,
                    "radius_mean",
                    "penalties",
                    f"Penalties {epoch}",
                    (230, 203, 0),
                    "dot",
                    (-0.1, self.cfg.interval.max_radius + 0.1),
                ),
                (
                    self.losses.total if self.losses else _,
                    "total_loss",
                    "loss",
                    f"Loss {epoch}",
                    (219, 0, 108),
                    "solid",
                    (-0.1, 35.0),
                ),
                (
                    self.losses.robust if self.losses else _,
                    "robust_loss",
                    "loss",
                    f"Loss {epoch}",
                    (7, 126, 143),
                    "solid",
                    (-0.1, 35.0),
                ),
            ],
        )

        # for layer, __ in self.model.named_interval_children():  # type: ignore
        #     metrics.append(
        #         (self.status.radius_mean_[layer] if self.status else _, f'radius_mean_{layer}',
        #             'penalties', f'Penalties {epoch}', (203, 203, 203), 'dash', (-0.1, 1.1))
        #     )

        return metrics

    def append_viz_debug(
            self,
            val: Tensor,
            name: str,
            window_name: str,
            color: tuple[int, int, int],
            dash: str,
            yrange: tuple[float, float],
    ):
        """Append single value to a Visdom line plot."""

        assert self.viz_debug

        if self.viz_reset_every_epoch:
            window_name = f"{window_name}_{(self.epoch or 0) + 1}"

        self.viz_debug.line(
            X=torch.tensor([self.mb_it]),
            Y=torch.tensor([val]),
            win=self.windows[window_name],
            update="append",
            name=name,
            opts={
                "linecolor": np.array([color]),  # type: ignore
                "dash": np.array([dash]),  # type: ignore
                "layoutopts": {
                    "plotly": {
                        "ytickmin": yrange[0],
                        "ytickmax": yrange[1],
                    }
                },
            },
        )

    def reset_viz_debug(self):
        """Recreate Visdom line plots before new epoch."""

        assert self.viz_debug

        for (
                _,
                name,
                window_name,
                title,
                color,
                dash,
                yrange,
        ) in self.get_debug_metrics():
            if self.viz_reset_every_epoch:
                window_name = f"{window_name}_{(self.epoch or 0) + 1}"

            # Reset plot line or create new plot
            self.windows[window_name] = self.viz_debug.line(
                X=torch.tensor([0]),
                Y=torch.tensor([0]),
                win=self.windows.get(window_name, None),
                opts={
                    "title": title,
                    "linecolor": np.array([color]),  # type: ignore
                    "dash": np.array([dash]),  # type: ignore
                    "layoutopts": {
                        "plotly": {
                            "margin": dict(l=40, r=40, b=80, t=80, pad=5),
                            "font": {"color": "rgb(0, 0, 0)"},
                            "legend": {"orientation": "h"},
                            "showlegend": True,
                            "yaxis": {"autorange": False, "range": yrange},
                        }
                    },
                },
                name=name,
            )
