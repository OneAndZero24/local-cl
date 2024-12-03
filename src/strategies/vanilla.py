from types import MethodType
from typing import Any, Optional, Sequence, Union

import torch
import torch.linalg
import torch.nn as nn
import wandb
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

from rich import print  # type: ignore # noqa
from torch import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer

from src.cfg import Settings
from src.models.dynamic import MultiTaskModule
from src.models.interval import Mode


class VanillaTraining(BaseStrategy):
    """Benchmark CL training."""

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
            *,
            cfg: Settings,
    ):
        super().__init__(  # type: ignore
            model,
            optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        # Avalanche typing specifications
        self.mb_it: int  # type: ignore
        self.mb_output: Tensor  # type: ignore
        self.loss: Tensor  # type: ignore
        self.training_exp_counter: int  # type: ignore
        self.optimizer: Optimizer
        self._criterion: CrossEntropyLoss
        self.device: torch.device

        # Config values
        self.cfg = cfg

        self.valid_classes = 0

        if self.cfg.offline is True:

            def train(
                    self,
                    experiences: Union[Experience, Sequence[Experience]],
                    eval_streams: Optional[Sequence[Union[Experience, Sequence[Experience]]]] = None,
                    **kwargs: Any,
            ):
                """Repurposed code from Avalanche."""
                self.is_training = True
                self.model.train()
                self.model.to(self.device)

                # Normalize training and eval data.
                if isinstance(experiences, Experience):
                    experiences = [experiences]
                if eval_streams is None:
                    eval_streams = [experiences]
                for i, exp in enumerate(eval_streams):
                    if isinstance(exp, Experience):
                        eval_streams[i] = [exp]  # type: ignore

                self._experiences = experiences
                self.before_training(**kwargs)  # type: ignore
                for exp in experiences:
                    self.train_exp(exp, eval_streams, **kwargs)  # type: ignore
                    # Joint training only needs a single step because
                    # it concatenates all the data at once.
                    break
                self.after_training(**kwargs)  # type: ignore

            self.train = MethodType(train, self)

            def train_dataset_adaptation(self, **kwargs: Any):
                self.adapted_dataset = AvalancheConcatDataset(
                    [exp.dataset for exp in self._experiences])  # type: ignore
                self.adapted_dataset = self.adapted_dataset.train()  # type: ignore

            self.train_dataset_adaptation = MethodType(train_dataset_adaptation, self)

    @property
    def mb_y(self) -> Tensor:
        """Current mini-batch target."""
        return super().mb_y  # type: ignore

    @property
    def mb_x(self) -> Tensor:
        """Current mini-batch."""
        return super().mb_x  # type: ignore

    @property
    def mb_task_id(self) -> Tensor:
        """Current mini-batch task labels."""
        return super().mb_task_id  # type: ignore

    def make_train_dataloader(self, num_workers=0, shuffle=True,
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

    def forward(self):
        if isinstance(self.model, MultiTaskModule):
            return self.model(self.mb_x, self.mb_task_id)
        else:  # no task labels
            return self.model(self.mb_x)

    def before_training_exp(self, **kwargs: Any):
        """Switch mode or freeze on each consecutive experience."""
        super().before_training_exp(**kwargs)  # type: ignore
        self.optimizer.param_groups[0]["lr"] = self.cfg.learning_rate

    def before_training_epoch(self, **kwargs: Any):
        """Switch to expansion phase when ready."""
        super().before_training_epoch(**kwargs)  # type: ignore
        # TODO hack - think of a better way to implement this
        if not hasattr(self, 'mode') or self.mode == Mode.VANILLA:
            if self.cfg.milestones is not None and self.epoch in self.cfg.milestones.keys():
                current_lr = self.optimizer.param_groups[0]["lr"]
                new_lr = current_lr * self.cfg.milestones[self.epoch]
                self.optimizer.param_groups[0]["lr"] = new_lr

    def after_training_epoch(self, **kwargs: Any):
        super().after_training_epoch(**kwargs)
        wandb.log({'lr': self.optimizer.param_groups[0]["lr"]}, commit=False)
