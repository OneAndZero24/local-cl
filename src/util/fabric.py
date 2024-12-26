from hydra.utils import instantiate
from omegaconf import DictConfig

from lightning.fabric import Fabric


def setup_fabric(config: DictConfig) -> Fabric:
    """
    Sets up Fabric run based on config.
    """

    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric