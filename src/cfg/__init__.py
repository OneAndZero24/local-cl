from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict

import pytorch_yard
from omegaconf import MISSING


class DatasetType(Enum):
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR100 = auto()
    CIFAR10 = auto()
    # CELEBA = auto()


class ScenarioType(Enum):
    INC_TASK = auto()
    INC_DOMAIN = auto()
    INC_CLASS = auto()


class StrategyType(Enum):
    Naive = auto()
    Joint = auto()
    Interval = auto()
    EWC = auto()
    SI = auto()
    LWF = auto()
    L2 = auto()
    MAS = auto()


class OptimizerType(Enum):
    SGD = auto()
    ADAM = auto()


# Interval training settings
@dataclass
class IntervalSettings:
    bias: bool = True

    vanilla_loss_threshold: float = 0.03
    expansion_learning_rate: float = 0.01
    scale_learning_rate: float = 0.01

    radius_multiplier: float = 1.0
    max_radius: float = 1.0
    radius_exponent: float = 0.5

    robust_lambda: float = 0.002

    metric_lookback: int = 10
    robust_accuracy_threshold: float = 0.9

    normalize_shift: bool = False
    normalize_scale: bool = False
    scale_init: float = -5.0

    contraction_epochs: int = 10


# General experiment settings validation schema & default values
@dataclass
class Settings(pytorch_yard.Settings):
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    seed: int = 1234

    enable_visdom: bool = False
    visdom_reset_every_epoch: bool = False

    batch_size: int = 128
    epochs: int = MISSING
    learning_rate: float = MISSING
    momentum: Optional[float] = MISSING
    optimizer: OptimizerType = MISSING
    weight_decay: float = 0.0
    milestones: Optional[Dict[int, float]] = field(default_factory=lambda: {})
    act_fn: str = "relu"

    # ----------------------------------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------------------------------
    dataset: DatasetType = DatasetType.MNIST
    offline: bool = False

    # ----------------------------------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------------------------------
    batchnorm: bool = False

    # ----------------------------------------------------------------------------------------------
    # Continual learning setup
    # ----------------------------------------------------------------------------------------------
    scenario: ScenarioType = ScenarioType.INC_DOMAIN
    n_experiences: int = 5

    # ----------------------------------------------------------------------------------------------
    # Strategy settings
    # ----------------------------------------------------------------------------------------------
    strategy: StrategyType = MISSING
    ewc_lambda: Optional[float] = None
    ewc_mode: Optional[str] = None
    ewc_decay: Optional[float] = None
    si_lambda: Optional[float] = None
    lwf_alpha: Optional[float] = None
    lwf_temperature: Optional[float] = None

    # ----------------------------------------------------------------------------------------------
    # IntervalNet specific settings
    # ----------------------------------------------------------------------------------------------
    interval: IntervalSettings = IntervalSettings()
