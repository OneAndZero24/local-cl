import functools
from typing import Any, Iterable, Optional, Type, cast

import pytorch_yard
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_multi_dataset_generic_benchmark,
)
from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericScenarioStream
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.training.plugins.evaluation import EvaluationPlugin
from pytorch_yard import info, info_bold
from pytorch_yard.avalanche import incremental_domain
from pytorch_yard.avalanche.scenarios import incremental_class, incremental_task
from pytorch_yard.experiments.avalanche import AvalancheExperiment
from rich import print
import torch
from torch import Tensor
from torch.optim import SGD, AdamW

from src.cfg import (
    DatasetType,
    OptimizerType,
    ScenarioType,
    Settings,
    StrategyType,
)
from src.datasets import fashion_mnist, mnist, cifar10, cifar100
from src.metrics.basic import EvalAccuracy, TotalLoss, TrainAccuracy
from src.metrics.interval import interval_training_diagnostics
from src.loss.activation import ActivationEntropyPlugin
from src.models.interval import (IntervalAlexNet, IntervalMLP,
                                         IntervalModel)
from src.models.lenet import LeNet
from src.models.standard import AlexNet, MLP, MobileNet
from src.strategies import (
    EWCPlugin,
    L2Plugin,
    LwFPlugin,
    MASPlugin,
    SynapticIntelligencePlugin,
    VanillaTraining,
    IntervalTraining
)

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Experiment(AvalancheExperiment):
    def __init__(
            self,
            config_path: str,
            settings_cls: Type[Settings],
            settings_group: Optional[str] = None,
    ) -> None:
        super().__init__(config_path, settings_cls, settings_group=settings_group)

        self.cfg: Settings
        """Experiment config."""

        self.input_size: int
        """Model input size."""

        self.channels: int
        """Model input number of channels."""

        self.n_classes: int
        """Number of classes for each head."""

        self.n_heads: int
        """Number of model heads."""

        self.model_map = {
            "MLP": self._get_mlp_model,
            "CNN": self._get_cnn_model,
            "LENET": self._get_lenet_model
        }

    def entry(self, root_cfg: pytorch_yard.RootConfig) -> None:
        super().entry(root_cfg)

    def main(self):
        super().main()

        self.setup_dataset()
        self.setup_scenario()

        if self.cfg.strategy is StrategyType.Naive:
            self.setup_naive()
        elif self.cfg.strategy is StrategyType.EWC:
            self.setup_ewc()
        elif self.cfg.strategy is StrategyType.L2:
            self.setup_l2()
        elif self.cfg.strategy is StrategyType.SI:
            self.setup_si()
        elif self.cfg.strategy is StrategyType.LWF:
            self.setup_lwf()
        elif self.cfg.strategy is StrategyType.MAS:
            self.setup_mas()
        elif self.cfg.strategy is StrategyType.Interval:
            self.setup_interval()
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.strategy}")

        print(self.model)

        self.setup_optimizer()
        self.setup_evaluator()
        self.setup_strategy()

        # ------------------------------------------------------------------------------------------
        # Experiment loop
        # ------------------------------------------------------------------------------------------
        info_bold("Starting experiment...")
        for i, experience in enumerate(cast(Iterable[NCExperience], self.scenario.train_stream)):
            info(f"Start of experience: {experience.current_experience}")
            info(f"Current classes: {experience.classes_in_this_experience}")

            self.strategy.valid_classes = len(experience.classes_seen_so_far)

            if self.cfg.offline is True:
                i = len(self.scenario.train_stream) - 1  # test on all data
                self.strategy.valid_classes = self.scenario.n_classes

            seen_datasets: list[AvalancheDataset[Tensor, int]] = [
                AvalancheDataset(exp.dataset, task_labels=t if self.cfg.scenario is ScenarioType.INC_TASK else 0)
                # type: ignore # noqa
                for t, exp in enumerate(self.scenario.test_stream[0: i + 1])  # type: ignore
            ]
            seen_test = functools.reduce(lambda a, b: a + b, seen_datasets)  # type: ignore
            seen_test_stream: GenericScenarioStream[Any, Any] = create_multi_dataset_generic_benchmark(
                [], [], other_streams_datasets={"seen_test": [seen_test]}
            ).seen_test_stream  # type: ignore

            if self.cfg.offline is True:
                self.strategy.train(self.scenario.train_stream,
                                    [self.scenario.test_stream, seen_test_stream])  # type: ignore
                break  # only one valid experience in joint training
            else:
                self.strategy.train(experience, [self.scenario.test_stream, seen_test_stream])  # type: ignore

            info("Training completed")

    # ------------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------------
    def setup_dataset(self):
        if self.cfg.dataset is DatasetType.MNIST:
            self.train, self.test, self.train_transform, self.eval_transform = mnist()
            self.n_classes = 10
            self.input_size = 28
            self.channels = 1
        elif self.cfg.dataset is DatasetType.CIFAR100:
            self.train, self.test, self.train_transform, self.eval_transform = cifar100()
            self.n_classes = 100
            self.input_size = 32
            self.channels = 3
        elif self.cfg.dataset is DatasetType.CIFAR10:
            self.train, self.test, self.train_transform, self.eval_transform = cifar10()
            self.n_classes = 10
            self.input_size = 32
            self.channels = 3
            self.input_size = 32
        elif self.cfg.dataset is DatasetType.FASHION_MNIST:
            self.train, self.test, self.train_transform, self.eval_transform = fashion_mnist()
            self.n_classes = 10
            self.input_size = 28
            self.channels = 1
        else:
            raise ValueError(f"Unknown dataset type: {self.cfg.dataset}")

    def setup_scenario(self):
        if self.cfg.scenario is ScenarioType.INC_TASK:
            _setup = incremental_task
            self.n_heads = self.cfg.n_experiences
        elif self.cfg.scenario is ScenarioType.INC_DOMAIN:
            _setup = incremental_domain
            self.n_heads = 1
        elif self.cfg.scenario is ScenarioType.INC_CLASS:
            _setup = incremental_class
            self.n_heads = 1
        else:
            raise ValueError(f"Unknown scenario type: {self.cfg.scenario}")

        self.scenario, self.n_classes_per_head = _setup(
            self.train,
            self.test,
            self.train_transform,
            self.eval_transform,
            self.cfg.n_experiences,
            self.n_classes,
        )

    def setup_optimizer(self):
        if self.cfg.optimizer is OptimizerType.SGD:
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                momentum=self.cfg.momentum if self.cfg.momentum else 0,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer is OptimizerType.ADAM:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=self.cfg.learning_rate,
                                   weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer}")

        print(self.optimizer)

    def setup_evaluator(self):
        metrics: list[PluginMetric[Any]] = [
            TotalLoss(),
            TrainAccuracy(),
            EvalAccuracy(),
        ]

        if self.cfg.strategy is StrategyType.Interval:
            assert isinstance(self.model, IntervalModel)
            metrics += interval_training_diagnostics(self.model)

        self.evaluator = EvaluationPlugin(
            *metrics,
            benchmark=self.scenario,
            loggers=[
                # TODO add a flag for optional RichLogger
                # RichLogger(
                #     ignore_metrics=[
                #         r"Diagnostics/(.*)",
                #         r"DiagnosticsHist/(.*)",
                #     ]
                # ),
                self.wandb_logger,
            ],
        )

    def setup_strategy(self):
        self.strategy = self.strategy_(
            model=self.model,
            optimizer=self.optimizer,
            train_mb_size=self.cfg.batch_size,
            train_epochs=self.cfg.epochs,
            eval_mb_size=self.cfg.batch_size,
            evaluator=self.evaluator,
            device=self.device,
            eval_every=1,
            cfg=self.cfg,
        )
        print(self.strategy)

    # ------------------------------------------------------------------------------------------
    # Experiment variants
    # ------------------------------------------------------------------------------------------
    def _get_lenet_model(self):
        return LeNet(
            self.cfg.model.size,
            self.cfg.model.stride,
            self.cfg.model.sizes,
            self.cfg.model.head_size,
            self.cfg.model.conv_type,
            self.cfg.model.head_type,
            self.cfg.model.add_avg_pool,
            self.cfg.model.initial_out_features,
            train_domain=self.cfg.model.train_domain,
            toggle_linear=self.cfg.model.toggle_linear
        )

    def _get_mlp_model(self):
        return MLP(
            input_size=self.input_size ** 2 * self.channels,
            hidden_dim=400,
            output_classes=self.n_classes_per_head,
            heads=self.n_heads,
        )

    def _get_cnn_model(self):
        return AlexNet(in_channels=3,
                       output_classes=self.n_classes_per_head,
                       heads=self.n_heads)

        # return VGG(
        #     variant='A',
        #     in_channels=3,
        #     output_classes=self.n_classes,
        #     heads=self.n_heads,
        #     batch_norm=self.cfg.batchnorm,
        # )

    def setup_naive(self):
        self.model = self.model_map[self.cfg.model]()
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[ActivationEntropyPlugin(self.cfg.gamma)]
        )

    def setup_ewc(self):
        self.model = self.model_map[self.cfg.model]()
        assert self.cfg.ewc_lambda and self.cfg.ewc_mode
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[EWCPlugin(self.cfg.ewc_lambda, self.cfg.ewc_mode, self.cfg.ewc_decay),
                     ActivationEntropyPlugin(self.cfg.gamma)
                ],
        )

    def setup_l2(self):
        self.model = self.model_map[self.cfg.model]()
        assert self.cfg.ewc_lambda and self.cfg.ewc_mode
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[L2Plugin(self.cfg.ewc_lambda, self.cfg.ewc_mode, self.cfg.ewc_decay),
                     ActivationEntropyPlugin(self.cfg.gamma)
                ],
        )

    def setup_si(self):
        self.model = self.model_map[self.cfg.model]()
        assert self.cfg.si_lambda
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[SynapticIntelligencePlugin(self.cfg.si_lambda),
                     ActivationEntropyPlugin(self.cfg.gamma)
                ],
        )

    def setup_lwf(self):
        self.model = self.model_map[self.cfg.model]()
        assert self.cfg.lwf_alpha and self.cfg.lwf_temperature
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[LwFPlugin(self.cfg.lwf_alpha, self.cfg.lwf_temperature),
                     ActivationEntropyPlugin(self.cfg.gamma)
                ],
        )

    def setup_mas(self):
        self.model = self.model_map[self.cfg.model]()
        assert self.cfg.ewc_lambda and self.cfg.ewc_mode
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[MASPlugin(self.cfg.ewc_lambda, self.cfg.ewc_mode, self.cfg.ewc_decay),
                     ActivationEntropyPlugin(self.cfg.gamma)
                ],
        )

    def setup_interval(self):
        if self.cfg.dataset is DatasetType.MNIST or self.cfg.dataset is DatasetType.FASHION_MNIST:
            self.model = IntervalMLP(
                input_size=self.input_size ** 2 * self.channels,
                hidden_dim=400,
                output_classes=self.n_classes_per_head,
                radius_multiplier=self.cfg.interval.radius_multiplier,
                max_radius=self.cfg.interval.max_radius,
                bias=self.cfg.interval.bias,
                heads=self.n_heads,
                normalize_shift=self.cfg.interval.normalize_shift,
                normalize_scale=self.cfg.interval.normalize_scale,
                scale_init=self.cfg.interval.scale_init,
            )
        elif self.cfg.dataset is DatasetType.CIFAR100 or self.cfg.dataset is DatasetType.CIFAR10:
            # self.model = IntervalSimpleCNN(
            #     input_size=self.input_size ** 2 * self.channels,
            #     hidden_dim=400,
            #     output_classes=self.n_classes_per_head,
            #     radius_multiplier=self.cfg.interval.radius_multiplier,
            #     max_radius=self.cfg.interval.max_radius,
            #     bias=self.cfg.interval.bias,
            #     heads=self.n_heads,
            #     normalize_shift=self.cfg.interval.normalize_shift,
            #     normalize_scale=self.cfg.interval.normalize_scale,
            #     scale_init=self.cfg.interval.scale_init,
            # )
            self.model = IntervalAlexNet(
                in_channels=self.channels,
                output_classes=self.n_classes_per_head,
                radius_multiplier=self.cfg.interval.radius_multiplier,
                max_radius=self.cfg.interval.max_radius,
                heads=self.n_heads,
                normalize_shift=self.cfg.interval.normalize_shift,
                normalize_scale=self.cfg.interval.normalize_scale,
                scale_init=self.cfg.interval.scale_init,
                act_fn=self.cfg.act_fn)
        self.strategy_ = functools.partial(
            IntervalTraining,
            enable_visdom=self.cfg.enable_visdom,
            visdom_reset_every_epoch=self.cfg.visdom_reset_every_epoch,
        )


if __name__ == "__main__":
    Experiment("src", Settings)
