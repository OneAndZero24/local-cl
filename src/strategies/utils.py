import torch.nn as nn
from torch import Tensor

from src.models.dynamic import MultiTaskModule


def avalanche_forward(model: nn.Module, x: Tensor, task_labels: Tensor):
    if isinstance(model, MultiTaskModule):
        return model(x, task_labels)
    else:  # no task labels
        return model(x)
