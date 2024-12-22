from hydra.utils import instantiate
from omegaconf import DictConfig
import fabric


def setup_fabric(config: DictConfig) -> fabric.Fabric:
    """
    Sets up Fabric run based on config.
    """

    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric