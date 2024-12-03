from collections import defaultdict
from typing import Any, Optional

import avalanche.training.plugins
import torch
import torch.nn as nn
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.utils import zerolike_params_dict
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .utils import avalanche_forward


class EWCPlugin(avalanche.training.plugins.ewc.EWCPlugin):
    """
    Hotfix for Avalanche EWC implementation.
    """

    def __init__(
        self,
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
    ):

        super().__init__(ewc_lambda, mode, decay_factor, keep_importance_data)

        self.mode: str
        self.saved_params: defaultdict[int, list[tuple[str, Tensor]]]
        self.importances: defaultdict[int, list[tuple[str, Tensor]]]

    def before_backward(self, strategy: BaseStrategy, **kwargs: Any):  # type: ignore
        """
        Compute EWC penalty and add it to the loss.
        """

        if strategy.training_exp_counter == 0:  # type: ignore
            return

        penalty = torch.tensor(0).float().to(strategy.device)  # type: ignore

        if self.mode == "separate":
            for experience in range(strategy.training_exp_counter):  # type: ignore
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(), self.saved_params[experience], self.importances[experience]
                ):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == "online":
            prev_exp: int = strategy.training_exp_counter - 1  # type: ignore # Cherry-pick #730
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                strategy.model.named_parameters(),
                self.saved_params[prev_exp],  # Cherry-pick #730
                self.importances[prev_exp],  # Cherry-pick #730
            ):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty  # type: ignore

    def compute_importances(
        self,
        model: nn.Module,
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
        dataset: AvalancheDataset[Any, Any],
        device: torch.device,
        batch_size: int,
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # list of list
        importances: list[tuple[str, Tensor]] = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for _, (x, y, task_labels) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out: Tensor = avalanche_forward(model, x, task_labels)
            ind = out.max(1)[1].flatten()  # Choose max prediction
            loss = criterion(out, ind)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp = imp / float(len(dataloader))

        return importances


class L2Plugin(EWCPlugin):
    def compute_importances(
        self,
        model: nn.Module,
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
        dataset: AvalancheDataset[Any, Any],
        device: torch.device,
        batch_size: int,
    ):
        """
        Return identity importance matrix (for L2 distance)
        """

        # list of list
        model.eval()

        # list of list
        importances: list[tuple[str, Tensor]] = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for _, (x, y, task_labels) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out: Tensor = avalanche_forward(model, x, task_labels)
            ind = out.max(1)[1].flatten()  # Choose max prediction
            loss = criterion(out, ind)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += torch.ones_like(p)
            break

        return importances


class MASPlugin(EWCPlugin):
    def compute_importances(
        self,
        model: nn.Module,
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
        dataset: AvalancheDataset[Any, Any],
        device: torch.device,
        batch_size: int,
    ):
        """
        Memory aware synapses.
        """

        model.eval()

        # list of list
        importances: list[tuple[str, Tensor]] = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for _, (x, y, task_labels) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out: Tensor = avalanche_forward(model, x, task_labels)

            loss = out.pow(2).mean()
            loss.backward()  # type: ignore

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().abs()

        # average over mini batch length
        for _, imp in importances:
            imp = imp / float(len(dataloader))

        return importances
